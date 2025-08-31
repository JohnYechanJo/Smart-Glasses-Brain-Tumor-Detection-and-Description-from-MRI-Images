
<div align="center">
<img src="./Smart Glasses for Brain Tumor Detection and Description from MRI Image.png" alt="Smart Glasses Brain Tumor Detection">
<br>
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=4&height=80">
<h1>Smart Glasses for Brain Tumor Detection and Description from MRI Images</h1>

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub last commit](https://img.shields.io/github/last-commit/JohnYechanJo/Smart-Glasses-Brain-Tumor-Detection-and-Description-from-MRI-Images)](https://github.com/JohnYechanJo/Smart-Glasses-Brain-Tumor-Detection-and-Description-from-MRI-Images/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/JohnYechanJo/Smart-Glasses-Brain-Tumor-Detection-and-Description-from-MRI-Images)](https://github.com/JohnYechanJo/Smart-Glasses-Brain-Tumor-Detection-and-Description-from-MRI-Images/issues)

## Contributors
[![GitHub Contributors](https://img.shields.io/github/contributors-anon/JohnYechanJo/Smart-Glasses-Brain-Tumor-Detection-and-Description-from-MRI-Images)](https://github.com/JohnYechanJo/Smart-Glasses-Brain-Tumor-Detection-and-Description-from-MRI-Images/graphs/contributors)
<table>
  <tr>
    <td align="center"><a href="https://github.com/JohnYechanJo"><img src="https://avatars.githubusercontent.com/u/131790222?v=4" width="100px;" alt=""/><br /><sub><b>John Yechan Jo</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/alicesaito2004"><img src="https://avatars.githubusercontent.com/u/229644208?v=4" width="100px;" alt=""/><br /><sub><b>Alice Saito</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/ZengKai"><img src="https://avatars.githubusercontent.com/u/placeholder_zengkai" width="100px;" alt=""/><br /><sub><b>Zeng Kai</b></sub></a><br /></td>
  </tr>
</table>
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=4&height=80&section=footer&fontSize=80">

</div>

## Overview
This project introduces lightweight smart glasses (68g, 9.6W power, 4.7-hour battery life) for real-time brain tumor detection and description from MRI images using **YOLOv11** and the **Grok3 API**. Designed for medical settings, it enhances radiology diagnostics and surgical guidance through augmented reality (AR) overlays, integrating computer vision and natural language processing for hands-free operation.

## Key Features
- **YOLOv11**: Ultralytics’ latest model achieves **99.48 mAP@0.5** on the Br35H dataset with **10-15 FPS** for real-time tumor detection.
- **Grok3 API**: xAI’s advanced LLM generates detailed tumor descriptions (size, location, risk) in **<100ms**.
- **Hardware**: Lightweight smart glasses with a micro OLED display and Wi-Fi connectivity for AR visualization.
- **AR Integration**: Hands-free display of tumor locations and surrounding structures for enhanced surgical precision.
- **Dataset**: Br35H brain MRI dataset for training and validation.
- **Implementation**: Combines YOLOv11, Grok3 API, and AR hardware, validated on smartphones (smart glass integration pending).

## Methodology
- **YOLOv11-Large**: Trained on the Br35H dataset for high-accuracy tumor detection in MRI images.
- **Grok3 API**: Processes detection outputs to generate natural language descriptions of tumor characteristics.
- **Hardware Design**: Integrates a micro OLED display, Wi-Fi module, and low-power components for real-time AR overlays.
- **Usage**:
  1. Upload an MRI image or capture via webcam.
  2. Analyze the image to view detection results.
  3. Review tumor areas via AR overlays on the smart glasses.

## Challenges
- **Battery Life**: 4.7-hour battery may be insufficient for extended surgeries.
- **Complex Cases**: AI may misinterpret rare or ambiguous tumors, risking false positives/negatives.
- **Wi-Fi Dependency**: Requires stable connectivity for real-time processing.
- **Display**: Micro OLED may lack detail for certain imaging needs.
- **Comfort**: Glasses may feel bulky during prolonged use.

## Results
| Model/Component | Metric | Value |
|-----------------|--------|-------|
| YOLOv11         | mAP@0.5 | 99.48 |
| YOLOv11         | FPS     | 10-15 |
| Grok3 API       | Response Time | <100ms |
| Hardware        | Weight | 68g |
| Hardware        | Power Consumption | 9.6W |
| Hardware        | Battery Life | 4.7 hours |

The system achieves high-accuracy tumor detection and rapid description generation, with lightweight hardware suitable for medical settings.

## Installation
```bash
# Clone the repository
git clone https://github.com/JohnYechanJo/Smart-Glasses-Brain-Tumor-Detection-and-Description-from-MRI-Images.git
cd Smart-Glasses-Brain-Tumor-Detection-and-Description-from-MRI-Images
