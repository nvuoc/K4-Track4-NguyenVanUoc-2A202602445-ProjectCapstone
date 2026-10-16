"""Rule-based temporal decision layer for a VinFast DMS hackathon demo.

FaceLandmarker performs pretrained face landmark & blendshape estimation.
This module interprets geometry and temporal mouth motion; it does NOT
prove audio speech, visual attention, or fitness for a real vehicle.
"""
from collections import deque
import numpy as np


class DriverMonitor:
    def __init__(self, mouth_variation_threshold=0.085, yaw_threshold=0.22,
                 pitch_threshold=0.16, combo_seconds=0.8, away_seconds=2.0):
        self.mouth_variation_threshold = float(mouth_variation_threshold)
        self.yaw_threshold = float(yaw_threshold)
        self.pitch_threshold = float(pitch_threshold)
        self.combo_seconds = float(combo_seconds)
        self.away_seconds = float(away_seconds)
        self.episodes = 0
        self.last_alert = False
        self.reset_calibration()

    def reset_calibration(self):
        self.calibration = []
        self.reference = None
        self.mouth_history = deque(maxlen=120)
        self.away_since = None
        self.combo_since = None
        self.last_alert = False

    def analyze(self, landmarks, blendshapes, timestamp_ms):
        """Accept one face's MediaPipe normalized landmarks (or None).

        blendshapes: MediaPipe Category objects with category_name / score.
        timestamp_ms: monotonic timestamp in milliseconds supplied by browser.
        """
        t = timestamp_ms / 1000.0
        base = dict(state='NO_FACE', speaking=False, head_away=False,
                    alert=False, calibrated=self.reference is not None,
                    yaw_delta=0.0, pitch_delta=0.0, mouth_signal=0.0,
                    mouth_variation=0.0, jaw_open=0.0, mar=0.0,
                    bbox=None, points=[], episodes=self.episodes)
        if not landmarks:
            self.mouth_history.clear()
            self.away_since = self.combo_since = None
            self.last_alert = False
            return base

        def pt(i):
            return np.array([landmarks[i].x, landmarks[i].y], dtype=float)

        eye_l, eye_r = pt(33), pt(263)
        eye_mid = (eye_l + eye_r) / 2.0
        nose, chin = pt(1), pt(152)
        eye_width = max(float(np.linalg.norm(eye_r-eye_l)), 1e-4)
        yaw_proxy = float(2.0 * (nose[0]-eye_mid[0]) / eye_width)
        pitch_proxy = float((nose[1]-eye_mid[1]) / max(abs(chin[1]-eye_mid[1]), 1e-4))
        mouth_width = max(float(np.linalg.norm(pt(61)-pt(291))), 1e-4)
        mar = float(np.linalg.norm(pt(13)-pt(14)) / mouth_width)
        shapes = {c.category_name: float(c.score) for c in (blendshapes or [])}
        jaw_open = float(shapes.get('jawOpen', 0.0))
        mouth_signal = float(0.5*min(1.0, 3.0*mar) + 0.5*jaw_open)

        xs = [max(0., min(1., float(p.x))) for p in landmarks]
        ys = [max(0., min(1., float(p.y))) for p in landmarks]
        base.update(mouth_signal=mouth_signal, jaw_open=jaw_open, mar=mar,
                    bbox=[min(xs), min(ys), max(xs), max(ys)],
                    points=[[float(landmarks[j].x), float(landmarks[j].y)]
                            for j in (33, 263, 1, 13, 14)])

        if self.reference is None:
            self.calibration.append((yaw_proxy, pitch_proxy))
            if len(self.calibration) >= 12:
                self.reference = np.median(np.array(self.calibration), axis=0)
                self.mouth_history.clear()
            base.update(state='CALIBRATING', calibrated=False)
            return base

        yaw_delta = abs(yaw_proxy - self.reference[0])
        pitch_delta = abs(pitch_proxy - self.reference[1])
        head_away = bool(yaw_delta > self.yaw_threshold or
                         pitch_delta > self.pitch_threshold)

        self.mouth_history.append((t, mouth_signal))
        while self.mouth_history and t-self.mouth_history[0][0] > 2.4:
            self.mouth_history.popleft()
        values = np.array([val for _, val in self.mouth_history])
        span = t-self.mouth_history[0][0] if self.mouth_history else 0.0
        variation = float(np.percentile(values, 90)-np.percentile(values, 10)) if len(values) >= 2 else 0.0
        motions = int(np.sum(np.abs(np.diff(values)) > 0.035)) if len(values) >= 2 else 0
        speaking = bool(len(values) >= 5 and span >= 0.8 and
                        variation >= self.mouth_variation_threshold and motions >= 2)

        if head_away:
            if self.away_since is None:
                self.away_since = t
        else:
            self.away_since = None
        if head_away and speaking:
            if self.combo_since is None:
                self.combo_since = t
        else:
            self.combo_since = None

        combo_alert = (self.combo_since is not None and t-self.combo_since >= self.combo_seconds)
        away_alert = (self.away_since is not None and t-self.away_since >= self.away_seconds)
        alert = bool(combo_alert or away_alert)
        if alert and not self.last_alert:
            self.episodes += 1
        self.last_alert = alert

        if combo_alert:
            state = 'ALERT_TALK_AWAY'
        elif away_alert:
            state = 'ALERT_AWAY'
        elif speaking and head_away:
            state = 'TALK_AWAY_PENDING'
        elif speaking:
            state = 'POSSIBLE_TALKING'
        elif head_away:
            state = 'LOOKING_AWAY'
        else:
            state = 'FOCUSED'
        base.update(state=state, speaking=speaking, head_away=head_away,
                    alert=alert, calibrated=True,
                    yaw_delta=round(float(yaw_delta), 4),
                    pitch_delta=round(float(pitch_delta), 4),
                    mouth_variation=round(variation, 4),
                    episodes=self.episodes)
        return base
