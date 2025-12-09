import argparse
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER


# ---------- KD pieces ----------
class KDLoss(nn.Module):
    def __init__(self, T=2.0):
        super().__init__()
        self.T = T
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits):
        T = self.T
        s = torch.log_softmax(student_logits / T, dim=-1)
        t = torch.softmax(teacher_logits / T, dim=-1)
        return self.kl(s, t) * (T * T)


def get_teacher_raw_preds(teacher_model, imgs):
    # try to get pre-NMS raw preds from DetectionModel
    if hasattr(teacher_model, "_predict_once"):
        return teacher_model._predict_once(imgs)
    # fallback
    return teacher_model(imgs)


# ---------- KD Trainer ----------
class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, teacher_path, kd_conf=0.3, kd_box=1.0, kd_cls=1.0, kd_temp=2.0, **kwargs):
        super().__init__(**kwargs)

        # load teacher
        LOGGER.info(f"[KD] Loading teacher from: {teacher_path}")
        teacher_yolo = YOLO(teacher_path)
        self.teacher = teacher_yolo.model.to(self.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # KD params
        self.kd_conf = kd_conf
        self.kd_box_w = kd_box
        self.kd_cls_w = kd_cls
        self.kd_cls_loss = KDLoss(T=kd_temp)
        self.l1 = nn.L1Loss(reduction="mean")

    def loss(self, batch, preds):
        """
        preds here is exactly what Ultralytics expects: usually (preds, feats)
        so we first call the parent to get the normal YOLO loss,
        then we add our distillation term.
        """
        base_loss_dict = super().loss(batch, preds)  # {'loss': ..., 'box_loss': ..., ...}

        # student preds for KD (first element if tuple/list)
        if isinstance(preds, (list, tuple)):
            s_preds = preds[0]
        else:
            s_preds = preds

        # get teacher preds (no grad)
        imgs = batch["img"].to(self.device, non_blocking=True).float() / 255.0
        with torch.no_grad():
            t_preds = get_teacher_raw_preds(self.teacher, imgs)

        # some versions may return non-tensor (e.g. list of Results) – then skip KD
        if not isinstance(s_preds, torch.Tensor) or not isinstance(t_preds, torch.Tensor):
            kd_loss = torch.tensor(0.0, device=self.device)
        elif s_preds.shape[-1] != t_preds.shape[-1]:
            # student and teacher output layouts differ – skip KD
            kd_loss = torch.tensor(0.0, device=self.device)
        else:
            # mask good teacher boxes
            t_conf = t_preds[..., 4]
            mask = t_conf > self.kd_conf
            if mask.sum() == 0:
                kd_loss = torch.tensor(0.0, device=self.device)
            else:
                # box
                s_box = s_preds[..., 0:4][mask]
                t_box = t_preds[..., 0:4][mask]
                box_kd = self.l1(s_box, t_box)

                # cls
                s_cls = s_preds[..., 5:][mask]
                t_cls = t_preds[..., 5:][mask]
                cls_kd = self.kd_cls_loss(s_cls, t_cls)

                kd_loss = self.kd_box_w * box_kd + self.kd_cls_w * cls_kd

        base_loss_dict["loss"] = base_loss_dict["loss"] + kd_loss
        # optionally log KD
        base_loss_dict["kd_loss"] = kd_loss.detach()
        return base_loss_dict


def main():
    parser = argparse.ArgumentParser("YOLOv8 KD trainer (teacher -> student)")
    parser.add_argument("--data", required=True, help="dataset yaml")
    parser.add_argument("--teacher", required=True, help="teacher weights, e.g. yolov8x.pt")
    parser.add_argument("--student", required=True, help="student weights, e.g. yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=0.01)
    # KD stuff
    parser.add_argument("--kd_conf", type=float, default=0.3)
    parser.add_argument("--kd_box", type=float, default=1.0)
    parser.add_argument("--kd_cls", type=float, default=1.0)
    parser.add_argument("--kd_temp", type=float, default=2.0)
    parser.add_argument("--project", type=str, default="runs")
    parser.add_argument("--name", type=str, default="kd_yolo")
    opt = parser.parse_args()

    overrides = dict(
        model=opt.student,
        data=opt.data,
        epochs=opt.epochs,
        imgsz=opt.imgsz,
        batch=opt.batch,
        workers=opt.workers,
        lr0=opt.lr0,
        project=opt.project,
        name=opt.name,
    )

    trainer = KDDetectionTrainer(
        teacher_path=opt.teacher,
        kd_conf=opt.kd_conf,
        kd_box=opt.kd_box,
        kd_cls=opt.kd_cls,
        kd_temp=opt.kd_temp,
        overrides=overrides,
    )
    trainer.train()


if __name__ == "__main__":
    main()

