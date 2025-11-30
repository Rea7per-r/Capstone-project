#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
import os
import csv
from collections import deque, defaultdict

import rospy
from std_msgs.msg import Float32MultiArray, Header

# -----------------------------
# Robust import of array message
# -----------------------------
def import_tracked_array_type():
    candidates = [
        ("ml_detector.msg", "TrackedObjectArray"),
        ("ml_detector.msg", "TrackedObjects"),
        ("vision_msgs.msg", "Detection2DArray"),
    ]
    for pkg, name in candidates:
        try:
            mod = __import__(pkg, fromlist=[name])
            return getattr(mod, name)
        except Exception:
            pass
    raise ImportError("Couldn't import a tracked-objects array message. Update candidates[] to your package/type.")

TrackedArrayMsg = import_tracked_array_type()

# ---------------- Torch model ----------------
import torch
import torch.nn as nn

class MultiStepLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=3,
                 dropout=0.3, bidirectional=True, horizon=10):
        super().__init__()
        self.horizon = horizon
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        dir_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * dir_factor, horizon * 2)  # (cx,cy)*H

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        if self.bidirectional:
            last = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            last = hn[-1]
        out = self.fc(last)
        return out.view(out.size(0), self.horizon, 2)

# ---------------- Helpers ----------------
def get_attr(obj, names, default=None):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            try:
                return float(v)
            except Exception:
                pass
    return default

def extract_bbox(track_obj):
    track_id = None
    for n in ["track_id", "id", "tid", "object_id"]:
        if hasattr(track_obj, n):
            try:
                track_id = int(getattr(track_obj, n))
                break
            except Exception:
                pass
    conf = get_attr(track_obj, ["conf", "confidence", "score"], 1.0)

    if hasattr(track_obj, "bbox"):
        b = getattr(track_obj, "bbox")
        x1 = get_attr(b, ["x1", "xmin", "left"])
        y1 = get_attr(b, ["y1", "ymin", "top"])
        x2 = get_attr(b, ["x2", "xmax", "right"])
        y2 = get_attr(b, ["y2", "ymax", "bottom"])
        if None not in (x1, y1, x2, y2):
            return (x1, y1, x2, y2), conf, track_id

    x1 = get_attr(track_obj, ["x1", "xmin", "left"])
    y1 = get_attr(track_obj, ["y1", "ymin", "top"])
    x2 = get_attr(track_obj, ["x2", "xmax", "right"])
    y2 = get_attr(track_obj, ["y2", "ymax", "bottom"])
    if None not in (x1, y1, x2, y2):
        return (x1, y1, x2, y2), conf, track_id

    if hasattr(track_obj, "results") and hasattr(track_obj, "bbox"):
        try:
            cx = float(track_obj.bbox.center.x)
            cy = float(track_obj.bbox.center.y)
            w  = float(track_obj.bbox.size_x)
            h  = float(track_obj.bbox.size_y)
            x1 = cx - w/2.0
            y1 = cy - h/2.0
            x2 = cx + w/2.0
            y2 = cy + h/2.0
            return (x1, y1, x2, y2), conf, track_id
        except Exception:
            pass

    return None, conf, track_id

def bbox_to_center_wh(x1, y1, x2, y2):
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w  = max(1e-6, x2 - x1)
    h  = max(1e-6, y2 - y1)
    return cx, cy, w, h

def center_wh_to_bbox(cx, cy, w, h):
    return [cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0]

# ---------------- Node ----------------
class PredictorNode:
    def __init__(self):
        self.seq_len    = int(rospy.get_param("~seq_len", 5))
        self.horizon_in = int(rospy.get_param("~horizon", 10))
        self.fixed_id   = int(rospy.get_param("~track_id", -1))
        self.csv_path   = rospy.get_param("~csv_path", "")
        self.topic_in   = rospy.get_param("~topic_in", "/tracked_objects")
        self.topic_out  = rospy.get_param("~topic_out", "/lstm_predictions")
        self.norm_w     = float(rospy.get_param("~norm_width", 640.0))
        self.norm_h     = float(rospy.get_param("~norm_height", 480.0))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size   = int(rospy.get_param("~input_size", 2))
        self.hidden_size  = int(rospy.get_param("~hidden_size", 128))
        self.num_layers   = int(rospy.get_param("~num_layers", 3))
        self.dropout      = float(rospy.get_param("~dropout", 0.3))
        self.bidirectional= bool(rospy.get_param("~bidirectional", True))

        self.model_path = rospy.get_param(
            "~lstm_model_path",
            os.path.expanduser("~/Downloads/lstm_modelmultipstep.pth")
        )
        if not os.path.exists(self.model_path):
            rospy.logerr(f"[LSTM] Model not found: {self.model_path}")
            raise SystemExit(1)

        # History buffers & state
        self.hist = defaultdict(lambda: deque(maxlen=self.seq_len))
        self.last_wh = {}
        self.last_conf = {}
        self.locked_id = None if self.fixed_id < 0 else self.fixed_id

        # IO
        self.pub = rospy.Publisher(self.topic_out, Float32MultiArray, queue_size=10)
        rospy.Subscriber(self.topic_in, TrackedArrayMsg, self.cb_tracks, queue_size=10)

        # Load model
        self.model, self.horizon = self._load_model_and_horizon(self.model_path)
        self.model.eval()
        rospy.loginfo(f"[LSTM] Model loaded. Effective horizon={self.horizon}, seq_len={self.seq_len}")

        # CSV logging
        self.csv_fp = None
        self.csv_writer = None
        if self.csv_path:
            out_dir = os.path.dirname(self.csv_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            self.csv_fp = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_fp)
            header = [
                "stamp","track_id",
                "obs_x1","obs_y1","obs_x2","obs_y2",
                "obs_cx","obs_cy","obs_w","obs_h","obs_conf",
            ]
            for k in range(1, self.horizon+1):
                header += [f"t+{k}_cx", f"t+{k}_cy", f"t+{k}_x1", f"t+{k}_y1", f"t+{k}_x2", f"t+{k}_y2"]
            self.csv_writer.writerow(header)
            self.csv_fp.flush()
            rospy.loginfo(f"[LSTM] Writing CSV to {self.csv_path}")

        rospy.on_shutdown(self._on_shutdown)

    def _on_shutdown(self):
        try:
            if self.csv_fp:
                self.csv_fp.close()
        except Exception:
            pass

    def _load_model_and_horizon(self, path):
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        ckpt_rows = state["fc.weight"].shape[0] if isinstance(state, dict) and "fc.weight" in state else None
        ckpt_h = int(ckpt_rows // 2) if ckpt_rows is not None else self.horizon_in
        H_eff = min(self.horizon_in, ckpt_h)
        model = MultiStepLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            horizon=H_eff
        ).to(self.device)

        # Trim head if needed
        if ckpt_rows is not None and state["fc.weight"].shape[0] != H_eff*2:
            state["fc.weight"] = state["fc.weight"][:H_eff*2, :].clone()
            state["fc.bias"] = state["fc.bias"][:H_eff*2].clone()
        try:
            model.load_state_dict(state, strict=False)
        except RuntimeError as e:
            rospy.logwarn(f"[LSTM] Warning loading state_dict: {e}")
        return model, H_eff

    @torch.no_grad()
    def cb_tracks(self, msg):
        objs = getattr(msg, "tracked_objects",
                       getattr(msg, "detections",
                               getattr(msg, "objects", [])))

        # ----------------- Modified: handle all tracks if _track_id=-1 -----------------
        if self.locked_id is None:
            valid_tracks = []
            for o in objs:
                bbox, conf, tid = extract_bbox(o)
                if bbox is not None and tid is not None:
                    valid_tracks.append((bbox, conf, tid))
        else:
            bbox, conf, tid = None, None, None
            for o in objs:
                bbox_tmp, conf_tmp, tid_tmp = extract_bbox(o)
                if tid_tmp == self.locked_id:
                    bbox, conf, tid = bbox_tmp, conf_tmp, tid_tmp
                    break
            valid_tracks = [(bbox, conf, tid)] if bbox is not None else []

        for bbox, conf, tid in valid_tracks:
            x1, y1, x2, y2 = bbox
            cx, cy, w, h = bbox_to_center_wh(x1, y1, x2, y2)
            self.last_wh[tid] = (w, h)
            self.last_conf[tid] = float(conf)

            cx_n, cy_n = cx / self.norm_w, cy / self.norm_h
            self.hist[tid].append([cx_n, cy_n])

            while len(self.hist[tid]) < self.seq_len:
                self.hist[tid].append(self.hist[tid][-1])

            seq = torch.tensor([list(self.hist[tid])], dtype=torch.float32, device=self.device)
            preds_norm = self.model(seq).detach().cpu().numpy()[0]

            preds = []
            lw, lh = self.last_wh.get(tid, (w, h))
            for k in range(self.horizon):
                cx_n_pred, cy_n_pred = float(preds_norm[k, 0]), float(preds_norm[k, 1])
                cx_p = cx_n_pred * self.norm_w
                cy_p = cy_n_pred * self.norm_h
                x1p, y1p, x2p, y2p = center_wh_to_bbox(cx_p, cy_p, lw, lh)
                preds.append([cx_p, cy_p, x1p, y1p, x2p, y2p])

            flat = []
            for i, p in enumerate(preds, 1):
                cxp, cyp, x1p, y1p, x2p, y2p = p
                flat += [float(tid), float(i), cxp, cyp, x1p, y1p, x2p, y2p]
            self.pub.publish(Float32MultiArray(data=flat))

            if self.csv_writer:
                stamp = msg.header.stamp.to_sec() if hasattr(msg, "header") and isinstance(msg.header, Header) else rospy.Time.now().to_sec()
                flat_preds = []
                for p in preds:
                    cxp, cyp, x1p, y1p, x2p, y2p = p
                    flat_preds += [cxp, cyp, x1p, y1p, x2p, y2p]
                self.csv_writer.writerow([
                    stamp, tid,
                    x1, y1, x2, y2,
                    cx, cy, w, h, self.last_conf.get(tid, 1.0),
                    *flat_preds
                ])
                self.csv_fp.flush()

def main():
    rospy.init_node("bytetracker_lstm_predictor")
    PredictorNode()
    rospy.spin()

if __name__ == "__main__":
    main()

