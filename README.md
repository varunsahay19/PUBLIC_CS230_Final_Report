(c) 2024 Patrick Nieman and Varun Sahay

Abstract:

Performance-based earthquake engineering relies on acceleration time series of real earthquakes to run dynamic simulations to assess the structural response of buildings. The limited number of available recordings from large-magnitude earthquakes makes challenging the process of selecting earthquake motions that accurately represent the nature of seismic hazard at a particular site. This project explores the use of machine learning techniques to generate realistic ground motion time histories based on earthquake magnitude and several other seismic source, site, and source-to-site path characteristics. A neural network based on long short-term memory (LSTM) cells is compared to a multi-stage deconvlution model; the latter architecture succeeded in predicting ground motions with response spectra, intensities, and durations close to those of the corresponding real records. These results suggest that with a larger training set and further refinement, such a model might one day serve as a substitute for traditional ground motion selection.




Notes on file organization:

Key loss functions used in both the Recurrent and Multi-stage models are filed in Multi-Stage Model/Losses and Architectures, which also contains architectures.py, where all model construction for the component models of the Multi-Stage model, as well as several experimental alternate architectures, are found.

The high-frequency and low-frequency versions of the oscillation model are accessed by changing the HF flag in corresponding scripts.

rockQueryv2.go, used in processing rock data geospatially, relies on a proprietary library of geometry analysis functions developed by Patrick Nieman for unrelated work; these library functions are not included here.
