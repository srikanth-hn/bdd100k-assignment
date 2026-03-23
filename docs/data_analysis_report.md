# BDD100K Data Analysis — Findings and Observations

## Dataset Overview

- Training split: 1,288,405 objects across 70,000 images
- Validation split: 185,578 objects across 10,000 images
- 10 detection classes: car, traffic sign, traffic light, person, truck, bus, bike, rider, motor, train

---

## Class Distribution

| Class | Train count | Val count | % of train |
|-------|------------|-----------|------------|
| car | 714,121 | 102,540 | 55.4% |
| traffic sign | 239,961 | 34,915 | 18.6% |
| traffic light | 186,301 | 26,891 | 14.5% |
| person | 91,435 | 13,265 | 7.1% |
| truck | 30,012 | 4,247 | 2.3% |
| bus | 11,688 | 1,597 | 0.9% |
| bike | 7,227 | 1,007 | 0.6% |
| rider | 4,522 | 649 | 0.4% |
| motor | 3,002 | 452 | 0.2% |
| train | 136 | 15 | 0.011% |

**Key finding:** Severe class imbalance — car:train ratio is 5,250:1.
The `train` class has only 136 samples, making it nearly impossible
to learn from without specialised augmentation.

---

## Anomalies Detected

1. **Severe class imbalance** — `train` class is only 0.011% of training data
   (136 objects). The model will struggle to learn this class without
   copy-paste augmentation or synthetic oversampling.

2. **Bounding box area drift** — `train` class average area drops from
   38,625px² in train to 29,394px² in val (>20% drift). This means the
   model will see larger train objects during training but smaller ones
   at inference — a distribution mismatch.

---

## Weather Distribution

| Weather | Train % | Val % |
|---------|---------|-------|
| Clear | 50.8% | 50.7% |
| Overcast | 14.4% | 13.8% |
| Undefined | 12.5% | 12.6% |
| Snowy | 7.7% | 7.7% |
| Partly cloudy | 7.5% | 8.0% |
| Rainy | 6.9% | 7.1% |
| Foggy | 0.14% | 0.12% |

**Key finding:** Foggy conditions are severely underrepresented (0.14%).
A model trained on this data will likely fail in foggy conditions —
important for ADAS deployment in northern European markets.

---

## Time of Day Distribution

| Time | Train % | Val % |
|------|---------|-------|
| Daytime | 56.8% | 56.8% |
| Night | 35.6% | 35.0% |
| Dawn/Dusk | 7.5% | 8.1% |

**Key finding:** 35.6% night images — the model must handle low-light
conditions. This justifies the `hsv_v=0.5` brightness augmentation
in training.

---

## Spatial Distribution (Vanishing Point Analysis)

Object density is highest at the image centre (vanishing point region),
confirming typical forward-facing dashcam perspective. Objects become
sparse at image edges. This spatial bias means the model may underperform
on edge-of-frame detections — relevant for pedestrian safety at
intersections.

---

## Occlusion Analysis

| Class | Occlusion rate |
|-------|---------------|
| rider | 89.2% |
| bike | 83.8% |
| motor | 76.5% |
| car | 67.7% |
| bus | 65.5% |
| truck | 65.5% |
| train | 58.8% |
| person | 58.0% |
| traffic sign | 11.3% |
| traffic light | 3.2% |

**Key finding:** Overall occlusion rate is 47.3%. Rider class has 89.2%
occlusion — almost every rider annotation is partially hidden by their
vehicle. This directly motivated using `iou=0.6` (softer NMS) in training
to prevent suppression of valid overlapping detections.

---

## Bounding Box Size Analysis

| Size bucket | Train % | Val % |
|-------------|---------|-------|
| Small (<0.5% image area) | 80.4% | 80.7% |
| Medium (0.5-2%) | 11.8% | 11.7% |
| Large (>2%) | 7.8% | 7.6% |

**Key finding:** 80% of all objects are small. Traffic lights average
only 507px² mean area (262px² median). At imgsz=640, this becomes
approximately 3-4 pixels wide — at the detection limit. This directly
justified using `imgsz=1280` in full training.

---

## Scene Distribution

| Scene | Train % |
|-------|---------|
| City street | 69.1% |
| Highway | 19.7% |
| Residential | 10.4% |
| Parking lot | 0.4% |
| Tunnel | 0.07% |

**Key finding:** 69% city street + 19% highway = 88% driving scenes
with fixed orientation. This justified `degrees=0.0` and `flipud=0.0`
augmentation settings — rotating driving scene images would create
unrealistic training samples.

---

## Connection to Model Training Decisions

Every training hyperparameter was derived from these data findings:

| Data finding | Training decision |
|-------------|------------------|
| Car:Train = 5,250x imbalance | `copy_paste=0.5`, focal loss |
| 80% small objects | `imgsz=1280` |
| Traffic light avg area 507px² | `imgsz=1280` |
| Occlusion rate 47.3% | `iou=0.6` (softer NMS) |
| Rider occlusion 89.2% | `copy_paste=0.5` |
| 35.6% night images | `hsv_v=0.5` |
| 14% rainy/snowy | `hsv_s=0.7` |
| 69% city / 19% highway | `degrees=0.0`, `flipud=0.0` |
| BBox area drift >20% | `scale=0.5` |