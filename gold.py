import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data_loader import Dataset
from models import Encoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LOG_FORMAT = "[%(asctime)s][%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def _run_file_prefix(config):
    """
    Prefix for checkpoints / TSV. OpenKE (--dataset_path): dùng --dataset tag;
    benchmark gốc: conceptnet / atomic.
    """
    if getattr(config, "openke_custom", False):
        return config.dataset + "_" + config.model_name
    return config.dataset_name + "_" + config.model_name


def _state_dict_for_save(model):
    """Khi train bằng DataParallel, checkpoint có prefix 'module.'. Lưu weights Encoder thuần để load_test đúng."""
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def _load_encoder_weights(model, ckpt_path, device):
    """Load checkpoint: bỏ prefix 'module.' nếu file được lưu từ DataParallel."""
    state = torch.load(ckpt_path, map_location=torch.device(device))
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must be a state_dict dict, got %s" % type(state))
    if state:
        first_key = next(iter(state.keys()))
        if first_key.startswith("module."):
            state = {k[len("module.") :]: v for k, v in state.items()}
    model.load_state_dict(state)


def run_train(config, pretrained_entity_embeddings, pretrained_relation_embeddings):
    model = Encoder(
        config,
        config.rnn_input_size,
        config.rnn_hidden_size,
        config.rnn_num_layers,
        config.dropout,
        config.alpha,
        config.device,
        use_ptlm=True,
        pretrained_entity_embeddings=pretrained_entity_embeddings,
        pretrained_relation_embeddings=pretrained_relation_embeddings,
    )

    batch_size = config.batch_size
    if torch.cuda.device_count() > 1:
        logging.info("try to use {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(config.device)
    try:
        criterion = nn.MarginRankingLoss(margin=config.gama, reduction="none")
    except TypeError:
        criterion = nn.MarginRankingLoss(config.gama, reduce=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    data_loader = DataLoader(
        range(config.nr_train_samples),
        batch_size=batch_size,
        collate_fn=dataset.gen_train_batch,
        num_workers=0,
    )
    logging.info("batch size = {}".format(batch_size))
    all_sum = 0
    model_saved_path = _run_file_prefix(config) + ".ckpt"
    all_acc = 0
    for epoch in range(config.num_epochs):
        with tqdm(total=len(data_loader)) as TD:
            for idx, sample in tqdm(enumerate(data_loader)):
                TD.set_description("[epoch {}/{}]".format(epoch + 1, config.num_epochs))
                (
                    batch_h,
                    batch_r,
                    batch_t,
                    batch_rule_h,
                    batch_rule_r,
                    batch_rule_t,
                    batch_rule_c,
                    masks,
                    batch_size,
                ) = sample

                batch_h = torch.LongTensor(batch_h).to(config.device)
                batch_r = torch.LongTensor(batch_r).to(config.device)
                batch_t = torch.LongTensor(batch_t).to(config.device)

                batch_rule_h = torch.LongTensor(batch_rule_h).to(config.device)
                batch_rule_r = torch.LongTensor(batch_rule_r).to(config.device)
                batch_rule_t = torch.LongTensor(batch_rule_t).to(config.device)
                batch_rule_c = torch.FloatTensor(batch_rule_c).to(config.device)

                masks_sum = sum(masks)
                masks = torch.FloatTensor(masks).to(config.device)

                out, out_att, rules_repr = model(
                    batch_h,
                    batch_r,
                    batch_t,
                    batch_rule_h,
                    batch_rule_r,
                    batch_rule_t,
                    batch_rule_c,
                )

                out = out.reshape(batch_size, -1, 2 * 3 * config.rnn_hidden_size)
                out_att = out_att.reshape(
                    batch_size, -1, 2 * 3 * config.rnn_hidden_size
                )
                rules_repr = rules_repr.reshape(
                    batch_size, -1, config.rule_inst_cnt, 2 * 3 * config.rnn_hidden_size
                )
                batch_rule_c = batch_rule_c.reshape(
                    batch_size, -1, config.rule_inst_cnt
                )
                pos_h = out[:, 0, :]
                neg_hs = []
                neg_zs = []
                for i in range(config.neg_cnt):
                    neg_hs.append(out[:, i + 1, :])
                    neg_zs.append(out_att[:, 2 * i + 2, :])
                    neg_zs.append(out_att[:, 2 * i + 3, :])

                pos_z0 = out_att[:, 0, :]
                pos_z1 = out_att[:, 1, :]

                pos_loss = config.local_lambda * torch.norm(
                    pos_z0 - pos_z1, p=2, dim=1
                ) + config.global_lambda * (
                    torch.norm(
                        pos_h.unsqueeze(dim=1).repeat(1, config.rule_inst_cnt, 1)
                        - rules_repr[:, 0, :, :],
                        p=2,
                        dim=2,
                    )
                    * batch_rule_c[:, 0, :]
                ).mean(
                    dim=1
                )

                neg_loss = torch.zeros(batch_size).to(config.device)
                neg_losses = []
                for i in range(config.neg_cnt):
                    tmp_loss = config.local_lambda * torch.norm(
                        neg_zs[i * 2] - neg_zs[i * 2 + 1], p=2, dim=1
                    ) + config.global_lambda * (
                        torch.norm(
                            neg_hs[i]
                            .unsqueeze(dim=1)
                            .repeat(1, config.rule_inst_cnt, 1)
                            - rules_repr[:, i + 1, :, :],
                            p=2,
                            dim=2,
                        )
                        * batch_rule_c[:, i + 1, :]
                    ).mean(
                        dim=1
                    )
                    neg_losses.append(tmp_loss)

                y = -torch.ones(batch_size).to(config.device)

                losses = []
                for j in range(config.neg_cnt):
                    losses.append(
                        criterion(pos_loss, neg_losses[j], y).reshape(-1, batch_size)
                    )
                loss = losses[0].mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                TD.set_postfix(loss="{:.5f}".format(loss.item()))
                TD.update(1)

        logging.info("start saving model ...")
        torch.save(
            _state_dict_for_save(model),
            model_saved_path[:-5] + str(epoch + 1) + ".ckpt",
        )
        logging.info("model saved")
        logging.info("start evaluating")

        model.eval()

        if True:
            all_triple = []
            with torch.no_grad():
                test_data_loader = DataLoader(
                    range(config.nr_train_samples),
                    batch_size=config.batch_size,
                    collate_fn=dataset.gen_test_batch,
                    num_workers=0,
                )
                for sample in tqdm(test_data_loader):
                    (
                        batch_h,
                        batch_r,
                        batch_t,
                        batch_rule_h,
                        batch_rule_r,
                        batch_rule_t,
                        batch_rule_c,
                        labels,
                        org,
                        masks,
                        batch_size,
                    ) = sample
                    batch_h = torch.LongTensor(batch_h).to(config.device)
                    batch_t = torch.LongTensor(batch_t).to(config.device)
                    batch_r = torch.LongTensor(batch_r).to(config.device)
                    masks = torch.FloatTensor(masks).to(config.device)
                    batch_rule_h = torch.LongTensor(batch_rule_h).to(config.device)
                    batch_rule_r = torch.LongTensor(batch_rule_r).to(config.device)
                    batch_rule_t = torch.LongTensor(batch_rule_t).to(config.device)
                    batch_rule_c = torch.FloatTensor(batch_rule_c).to(config.device)
                    out, out_att, rules_repr = model(
                        batch_h,
                        batch_r,
                        batch_t,
                        batch_rule_h,
                        batch_rule_r,
                        batch_rule_t,
                        batch_rule_c,
                    )
                    out_att = out_att.reshape(
                        batch_size, 2, 2 * 3 * config.rnn_hidden_size
                    )
                    out_att_view0 = out_att[:, 0, :]
                    out_att_view1 = out_att[:, 1, :]

                    rules_repr = rules_repr.reshape(
                        batch_size,
                        -1,
                        config.rule_inst_cnt,
                        2 * 3 * config.rnn_hidden_size,
                    )
                    batch_rule_c = batch_rule_c.reshape(
                        batch_size, -1, config.rule_inst_cnt
                    )
                    loss = config.local_lambda * torch.norm(
                        out_att_view0 - out_att_view1, p=2, dim=1
                    ) + config.global_lambda * (
                        torch.norm(
                            out.unsqueeze(dim=1).repeat(1, config.rule_inst_cnt, 1)
                            - rules_repr[:, 0, :, :],
                            p=2,
                            dim=2,
                        )
                        * batch_rule_c[:, 0, :]
                    ).mean(
                        dim=1
                    )

                    loss = loss * masks

                    loss = list(loss.to("cpu").numpy())
                    for j in range(len(labels)):
                        all_triple.append([org[j], labels[j], loss[j]])

            all_triple.sort(key=lambda x: -x[2])
            tloss = []
            tlabel = []
            for j in all_triple:
                tloss.append(j[2])
                tlabel.append(j[1])
            try:
                auc = metrics.roc_auc_score(tlabel, tloss)
            except ValueError:
                auc = float("nan")
            if config.nr_error > 0:
                crr = 0
                for j in range(min(config.nr_error, len(all_triple))):
                    crr += all_triple[j][1]
                acc = crr / config.nr_error
            else:
                acc = float("nan")
            logging.info(
                "[ACC REPORT] Epoch {}, Acc = {:.5f}, AUC = {:.5f}".format(
                    epoch + 1, acc, auc
                )
            )
        model.train()


def run_test(
    config, pretrained_entity_embeddings, pretrained_relation_embeddings, train_list
):
    model_test = Encoder(
        config,
        config.rnn_input_size,
        config.rnn_hidden_size,
        config.rnn_num_layers,
        config.dropout,
        config.alpha,
        config.device,
        use_ptlm=True,
        pretrained_entity_embeddings=pretrained_entity_embeddings,
        pretrained_relation_embeddings=pretrained_relation_embeddings,
    )
    model_test.to(config.device)

    model_saved_path = _run_file_prefix(config) + str(config.num_epochs) + ".ckpt"
    logging.info("start loading model[{}] ...".format(model_saved_path))
    _load_encoder_weights(model_test, model_saved_path, config.device)
    logging.info("model loaded!")
    model_test.eval()

    data_loader = DataLoader(
        range(config.nr_train_samples),
        batch_size=config.batch_size,
        collate_fn=dataset.gen_test_batch,
        num_workers=0,
    )
    all_loss = []
    all_label = []

    result_file_name = _run_file_prefix(config)
    result_file_tsv = None
    if config.output_tsv:
        result_file_tsv = open(result_file_name + ".tsv", "w", encoding="utf-8")

    try:
        with torch.no_grad():
            for sample in tqdm(data_loader):
                (
                    batch_h,
                    batch_r,
                    batch_t,
                    batch_rule_h,
                    batch_rule_r,
                    batch_rule_t,
                    batch_rule_c,
                    labels,
                    org,
                    masks,
                    batch_size,
                ) = sample
                batch_h = torch.LongTensor(batch_h).to(config.device)
                batch_t = torch.LongTensor(batch_t).to(config.device)
                batch_r = torch.LongTensor(batch_r).to(config.device)
                batch_rule_h = torch.LongTensor(batch_rule_h).to(config.device)
                batch_rule_r = torch.LongTensor(batch_rule_r).to(config.device)
                batch_rule_t = torch.LongTensor(batch_rule_t).to(config.device)
                batch_rule_c = torch.FloatTensor(batch_rule_c).to(config.device)
                out, out_att, rules_repr = model_test(
                    batch_h,
                    batch_r,
                    batch_t,
                    batch_rule_h,
                    batch_rule_r,
                    batch_rule_t,
                    batch_rule_c,
                )

                rules_repr = rules_repr.reshape(
                    batch_size,
                    -1,
                    config.rule_inst_cnt,
                    2 * 3 * config.rnn_hidden_size,
                )
                batch_rule_c = batch_rule_c.reshape(
                    batch_size, -1, config.rule_inst_cnt
                )
                out_att = out_att.reshape(
                    batch_size, 2, 2 * 3 * config.rnn_hidden_size
                )
                out_att_view0 = out_att[:, 0, :]
                out_att_view1 = out_att[:, 1, :]

                loss = config.local_lambda * torch.norm(
                    out_att_view0 - out_att_view1, p=2, dim=1
                ) + config.global_lambda * (
                    torch.norm(
                        out.unsqueeze(dim=1).repeat(1, config.rule_inst_cnt, 1)
                        - rules_repr[:, 0, :, :],
                        p=2,
                        dim=2,
                    )
                    * batch_rule_c[:, 0, :]
                ).mean(dim=1)

                all_loss += list(loss.to("cpu").numpy())
                all_label += labels

                if config.output_tsv:
                    for j in range(len(labels)):
                        h, r, t = train_list[org[j]][0]
                        l = labels[j]
                        s = loss[j].item()
                        print(
                            h,
                            r,
                            t,
                            dataset.entity[h],
                            dataset.relation[r],
                            dataset.entity[t],
                            l,
                            s,
                            sep="\t",
                            file=result_file_tsv,
                        )

        all_loss = np.array(all_loss)
        try:
            auc = metrics.roc_auc_score(all_label, all_loss)
        except ValueError:
            auc = float("nan")

        idx = list(range(len(all_loss)))
        idx.sort(key=lambda x: -all_loss[x])
        if config.nr_error > 0:
            acc = 0
            for i in range(min(config.nr_error, len(idx))):
                acc += all_label[idx[i]]
            acc = acc / float(config.nr_error)
        else:
            acc = float("nan")

        logging.info(
            "[TEST RESULT] Acc = {:.5f}, AUC = {:.5f}".format(
                acc,
                auc,
            )
        )
    finally:
        if result_file_tsv is not None:
            result_file_tsv.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="C-05",
        help="Benchmark id (C-05, A-10, ...) or custom tag for OpenKE export "
        "(must match errors/*.txt and rules/*.pkl prefix).",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="If set, use this folder as GOLD dataset (e.g. OpenKE export).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for training/testing.",
    )
    parser.add_argument(
        "--model_name", type=str, default="noname", help="The name of the model."
    )
    parser.add_argument(
        "--epoch", type=int, default=10, help="The number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="The batch size for training and inference.",
    )
    parser.add_argument(
        "--topk", type=int, default=100, help="The number of top rules to use."
    )
    parser.add_argument(
        "--ptlm_model",
        type=str,
        default="sentence-transformers/sentence-t5-xxl",
        help="The name of the pre-trained language model to be used.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="The learning rate for training the model.",
    )
    parser.add_argument(
        "--local_lambda",
        type=float,
        default=0.1,
        help="The value of the lambda_local parameter.",
    )
    parser.add_argument(
        "--global_lambda",
        type=float,
        default=0.01,
        help="The value of the lambda_global parameter.",
    )
    parser.add_argument(
        "--neg_cnt",
        type=int,
        default=1,
        help="The number of negative samples to use during training.",
    )
    parser.add_argument(
        "--seed", type=int, default=5, help="The random seed for reproducibility."
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Flag indicating whether to run in test mode only.",
    )
    parser.add_argument(
        "--not_use_ptlm",
        action="store_true",
        help="Flag indicating whether to disable the use of pre-trained language models.",
    )
    parser.add_argument(
        "--output_tsv",
        action="store_true",
        help="Flag indicating whether to output the results in TSV format.",
    )
    args = parser.parse_args()

    config = Config()
    config.ptlm_model = args.ptlm_model
    config.num_epochs = args.epoch
    config.model_name = args.model_name
    config.neg_cnt = args.neg_cnt
    config.local_lambda = args.local_lambda
    config.learning_rate = args.lr
    config.global_lambda = args.global_lambda
    config.dataset = args.dataset
    config.rule_top_k = args.topk
    config.seed = args.seed
    config.use_ptlm = not (args.not_use_ptlm)
    config.output_tsv = args.output_tsv

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if args.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available; using CPU.")
        config.device = "cpu"
    else:
        config.device = args.device

    if args.dataset_path:
        config.dataset_path = os.path.abspath(args.dataset_path)
        if not os.path.isdir(config.dataset_path):
            raise FileNotFoundError("dataset_path not found: %s" % config.dataset_path)
        config.dataset_name = "openke"
        config.dataset = args.dataset
        config.openke_custom = True
        config.error_ratio = 0.0
    else:
        config.openke_custom = False
        if config.dataset.startswith("C"):
            config.dataset_name = "conceptnet"
            config.dataset_path = "./dataset/conceptnet"
        elif config.dataset.startswith("A"):
            config.dataset_name = "atomic"
            config.dataset_path = "./dataset/atomic"
        else:
            raise ValueError(
                "Unknown --dataset %r. Use C-* / A-* or pass --dataset_path."
                % (config.dataset,)
            )
        try:
            config.error_ratio = int(config.dataset.split("-")[-1]) / 100.0
        except Exception:
            config.error_ratio = 0.0

    if not os.path.exists("log"):
        os.makedirs("log")
    file_handler = logging.FileHandler(
        os.path.join(
            "log/",
            config.dataset + "_" + config.model_name + "_log.txt",
        )
    )
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logging.info("only for test ? {}".format(args.test_only))
    dataset = Dataset(config=config)

    config.nr_train_samples, config.nr_error = dataset.read()

    (
        pretrained_entity_embeddings,
        pretrained_relation_embeddings,
    ) = dataset.get_pretrained_embedding()

    config.total_ent = dataset.entity.cnt
    config.total_rel = dataset.relation.cnt
    config.total_triple = dataset.triple.cnt
    config.batch_size = args.batch_size

    if args.test_only == False:
        original_batch_size = config.batch_size
        ngpu = torch.cuda.device_count()
        if ngpu > 0:
            config.batch_size *= ngpu
        run_train(config, pretrained_entity_embeddings, pretrained_relation_embeddings)
        config.batch_size = original_batch_size

    logging.info("start testing")
    run_test(
        config,
        pretrained_entity_embeddings,
        pretrained_relation_embeddings,
        dataset.train_list,
    )

    print("DONE")
