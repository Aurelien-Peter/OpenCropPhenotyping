# OpenCropPhenotyping

**An open-source Python framework for high-throughput crop phenotyping using drone and satellite imagery.**

## About the project
OpenCropPhenotyping is an open-source project dedicated to reproducible high-throughput crop phenotyping using remote sensing data.
The objective is to provide reproducible workflows to extract agronomic traits from UAV and Sentinel-2 imagery, from raw image preprocessing to trait extraction, statistical analysis and visualization. The first developments focus on major field crops such as maize, with the possibility of extending the framework to other crops in future releases.

## Current Features:
- [X] NDVI computation 
- [X] NDRE computation
- [X] GNDVI computation 
- [X] SAVI computation  
- [X] Sentinel images resampling
- [X] Vegetation masking 
- [X] Crop cover estimation
- [X] Statistical analysis 
- [X] Visualization

## Case studies

The project includes reproducible case studies based on real agricultural areas in Southern France.

Current case studies:

- Maize monitoring with Sentinel-2 (Occitanie)

## Roadmap

### Version 0.1
#### Sentinel-2

- Read Sentinel-2 imagery
- NDVI computation
- Visualization
- GeoTIFF export
- PNG export
- Toy dataset
- Unit tests

---

### Version 0.2
#### Vegetation indices

- NDRE computation
- SAVI computation
- GNDVI computation
- Generic vegetation index computation
- Statistics
- Vegetation masking
- Trait Extraction (Crop cover)
- Batch processing
- Command-line interface (CLI)

---

### Version 0.3
#### Drone imagery

- UAV imagery
- Orthomosaic support
- Vegetation segmentation
- RGB vegetation indices
- Image tiling

---

### Version 0.4
#### Machine Learning

- Feature extraction
- Dataset preparation
- Classical classifiers (Random Forest, SVM)
- Crop classification
- Model evaluation
- Model persistence

### Version 0.5
#### Deep Learning

- PyTorch support
- CNN semantic segmentation
- U-Net implementation
- Training pipeline
- Inference pipeline
- Model visualization

### Version 1.0

Complete crop phenotyping workflow

## For which users?
This project is intended for researchers, agronomists, engineers working in remote sensing, as well as for UAV practitioners and interested students. The project is also suitable for anyone wishing to learn how to process drone or Sentinel-2 imagery using Python.

## Installation

git clone ...

cd OpenCropPhenotyping

conda env create -f environment.yml

conda activate opencropphenotyping

pip install -e .

## How to use

The repository already includes a small Sentinel-2 toy dataset.

No additional data download is required to run the examples

## Example workflow

The following workflow illustrates the processing of Sentinel-2 multispectral imagery, from the input bands to vegetation indices, statistical analysis and crop trait extraction.

### Sentinel-2 input bands

#### Sentinel-2 Red band (B04)

![B04](docs/images/red.png)

#### Sentinel-2 Near Infra Red band (B08)

![B08](docs/images/nir.png)

#### Green band (B03)

![B03](docs/images/green.png)

#### Red Edge band (B05)

![B05](docs/images/red_edge.png)

↓

### Vegetation indices

The selected Sentinel-2 bands are combined to compute several vegetation indices:

B04 Red ─────────────┬──→ NDVI
                     └──→ SAVI
                         
B08 NIR ─────────────┬──→ NDVI
                     ├──→ SAVI
                     ├──→ GNDVI
                     └──→ NDRE

B03 Green ──────────────→ GNDVI

B05 Red Edge ───────────→ NDRE     

#### NDVI — Normalized Difference Vegetation Index

![NDVI](docs/images/ndvi.png)

#### NDRE — Normalized Difference Red Edge Index

![NDRE](docs/images/ndre.png)

#### GNDVI — Green Normalized Difference Vegetation Index

![GNDVI](docs/images/gndvi.png)

#### SAVI — Soil-Adjusted Vegetation Index

![SAVI](docs/images/savi.png)

↓

### Statistics and visualization

Statistics are computed for each vegetation index to characterize the distribution of pixel values.

#### NDVI distribution

![NDVI distribution](docs/images/ndvi_hist_0.png)

#### NDVI boxplot

![NDVI boxplot](docs/images/ndvi_hist_1.png)

↓

### Trait extraction

The NDVI image is used to create a vegetation mask and estimate crop cover.

#### Vegetation mask

![Vegetation mask](docs/images/vegetation_mask_custom.png)

Crop cover: **39.24%**

↓

### Export

## Contributing 
Contributions are welcome. If you have ideas for improvements, bug fixes or new features, feel free to open an issue or submit a pull request.

## Documentation
Documentation will be progressively added as the project evolves.

## Version
This project follows Semantic Versioning (SemVer). 
See the Releases page for available versions and change logs.

## LICENSE
See the LICENSE file.
