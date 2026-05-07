# DarkDriving-ICRA-2026
DarkDriving is the first real-world day–night aligned autonomous driving dataset using TTPM. It contains 9,538 pairs with centimeter-level alignment and annotations, enabling low-light enhancement and perception tasks. This repo provides key code and data.

## Dataset
A real-world paired day-night driving dataset with centimeter-level alignment for low-light enhancement and autonomous driving perception.
### News

- **[2026-03]** DarkDriving paper is available.
- **[Coming Soon]** Dataset download link will be released.

---

### Introduction

**DarkDriving** is a real-world day and night aligned dataset for autonomous driving in dark environments.  
It is designed to support research on low-light enhancement, day-night image translation, and perception robustness for autonomous driving.

Existing low-light enhancement datasets are usually collected in small static scenes by controlling exposure, while many nighttime driving datasets do not provide precisely aligned daytime counterparts. DarkDriving addresses this limitation by collecting real-world driving scenes with precisely aligned daytime and nighttime image pairs.

The dataset is collected in a large-scale closed driving test field using an automated vehicle. A Trajectory Tracking based Pose Matching method is used to obtain highly aligned day-night image pairs.


https://github.com/user-attachments/assets/finnal_video.mp4


---

### Highlights

- **The first real-world centimeter-level aligned day-night driving dataset** for autonomous driving in dark environments.
- **Real-world driving scenes** collected in a large closed driving test field.
- **9,538 day-night aligned image pairs**.
- **19,076 high-resolution RGB images**.
- **2448 × 2048 image resolution**.
- **2D bounding box annotations** for autonomous driving perception.
- **6 types of road scenes** and **12 types of nighttime lighting conditions**.
- Supports low-light enhancement, day-night image translation, 2D detection, and 3D detection.

---

### Dataset Overview

| Item | Description |
|---|---|
| Dataset Name | DarkDriving |
| Scene Type | Real-world autonomous driving |
| Sensor | Front-view RGB camera |
| Image Resolution | 2448 × 2048 |
| Number of Image Pairs | 9,538 |
| Number of RGB Images | 19,076 |
| Training Set | 5,906 pairs |
| Testing Set | 3,632 pairs |
| Annotation Type | 2D bounding boxes |
| Main Object Class | Car |
| Number of 2D Boxes | 13,184 |
| Road Scene Types | 6 |
| Nighttime Lighting Conditions | 12 |

---

### Dataset Comparison

DarkDriving is different from previous low-light enhancement datasets and nighttime driving datasets.  
Compared with exposure-controlled low-light datasets, DarkDriving focuses on real driving scenes.  
Compared with nighttime driving datasets with rough GPS alignment, DarkDriving provides precisely aligned day-night image pairs.

<p align="center">
  <img src="assets/LOL_dataset.png" width="45%">
  <img src="assets/DARK_dataset.png" width="45%">
</p>
<p align="center">
  Others
</p>

<p align="center">
  <img src="assets/our2.png" width="45%">
  <img src="assets/our1.png" width="45%">
</p>
<p align="center">
  Ours
</p>

<p align="center">
  <b>Comparison with representative low-light and nighttime driving datasets.</b>
</p>

---

### Data Collection

#### Collection Vehicle

DarkDriving is collected using an automated vehicle equipped with a front-view RGB camera, LiDAR, GPS, and IMU sensors.

<p align="center">
  <img src="assets/changan.png" width="45%">
  <img src="assets/无人车图片1.png" width="45%">
</p>

<p align="center">
  <b>Automated vehicle used for data collection.</b>
</p>

#### Sensor Setup

<p align="center">
  <img src="assets/ChanganCAVSENSOR.png" width="45%">
  <img src="assets/无人车传感器坐标.png" width="45%">
</p>

<p align="center">
  <b>Sensor setup and coordinate system.</b>
</p>

---

### Collection Site

The dataset is collected in a large-scale closed driving test field.  
The closed test field allows us to control static background vehicles, streetlights, and different nighttime lighting conditions.

<p align="center">
  <img src="assets/testfiled.png" width="90%">
</p>

<p align="center">
  <b>Large-scale closed driving test field and high-precision map.</b>
</p>

---

### Day-Night Alignment Method

To collect precisely aligned day-night image pairs, we use a **Trajectory Tracking based Pose Matching** pipeline.

The overall pipeline contains the following steps:

1. Build or use a high-precision point cloud map of the test field.
2. Record the desired driving trajectory.
3. Let the automated vehicle follow the same trajectory during daytime.
4. Let the automated vehicle follow the same trajectory during nighttime.
5. Match day and night frames according to vehicle poses.
6. Refine the matched pairs manually to remove mismatched dynamic objects and large alignment errors.

<p align="center">
  <img src="assets/framework.png" width="95%">
</p>

<p align="center">
  <b>Trajectory Tracking based Pose Matching pipeline for day-night aligned data collection.</b>
</p>

---

### Scenario Diversity

DarkDriving contains diverse road scenes and nighttime lighting conditions.

#### Road Scene Types

The dataset includes the following road scenes:

- Multi-lane road
- Single-lane road
- Curved road
- Open road
- T-intersection
- Intersection

#### Nighttime Lighting Conditions

The dataset contains various nighttime lighting conditions, including:

- No streetlight
- Vehicle low beam
- Vehicle high beam
- Vehicle backlight
- Bilateral streetlights
- Unilateral streetlights
- Streetlights with vehicle beams
- Streetlights with vehicle backlight

---
