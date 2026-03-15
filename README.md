# 2D Chest CT Organ Segmentation

[![Language](https://img.shields.io/badge/language-English%20%7C%20%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-blue.svg)](#)
[English] | [简体中文](./README_zh.md)

A deep learning project for 2D thoracic organ segmentation in CT scans. This repository supports training from scratch using custom images and masks.

## Key Features

- **Training from Scratch**: Full pipeline for training on your own CT datasets.
- **Data Processing**: 
  - Input: 256x256 grayscale images.
  - Built-in conversion: Automatically transforms RGB color masks into **Label Maps** for training.
- **Model Zoo**: Includes three architecture choices:
  - `U-Net`
  - `Simple_NestNet` (UNet++)
  - `NestNet with Backbone`

## Segmentation Results

Here is an example of the model's performance after training:

<div align="center">
  <img src="./doc/val_000_03_ID00007637202177411956430_11.png" width="600" alt="Segmentation Example">
  <p><i>Figure 1: Comparison between Original CT and Segmented Masks</i></p>
</div>

## Quick Start

1. **Prepare Data**: Place your 256x256 grayscale images and RGB masks in the data folder.
2. **Preprocessing**: Use the internal script to convert RGB masks to label maps.
3. **Train**: Choose your model and start training.
