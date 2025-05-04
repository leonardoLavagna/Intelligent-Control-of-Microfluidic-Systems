# Intelligent Control of Microfluidic Systems
This is the code for the article ... in which we apply machine learning techniques to [microfluidics-based](https://en.wikipedia.org/wiki/Microfluidics) [liposome](https://en.wikipedia.org/wiki/Liposome) production at laboratory scale. 


WORK IN PROGRESS...


## What's in here?
Here you can find all the processing steps we carried out.
- Underscore directories contain auxiliary files and should not be modified, in particular:
  - `_Logs` contains all the relevant logs, in particular the results of data cleaning (in `raw_data_checker.log` and `data_checker.log`), of the exploratory data analysis (in `exploratory_data_analysis.log`) and other tasks (data augmentation, data splitting and aggregation).
  - `_Files` contains configuration files and auxiliaries needed to carry out the preliminary data cleaning and feature engineering.
  - `_Models` contains the pickle files corresponding to the best pre-trained models we used.
  - `_Preprocessing` contains the functions needed to carry out the preliminary data cleaning and feature engineering.
  - `_Raw_Data` contains the raw data files as they have been generated in the lab. These datapoints are etherogeneous formulations, we focused on the formulations obtained via a microfluidic setup that involved a micromixer chip (see...).
- `Data` contains the cleaned and preprocessed datasets. In particular:
  - The file `seed.csv` contains the first batch of data that was originally prepared for...
  - The file `extension.csv` contains a second batch of experimental data originally prepared for...
  - The set `validation.csv` contains the wet-lab validation experiments used to benchmark the performances of the proposed models.
  - The file `data_1.csv` is the dataset obtained merging `seed.csv` and `extension.csv`. This is the main file we considered in the project.
  - The file `data_2.csv` is obtained from `data_1.csv` with a gaussian-noise-data-extension (see the `data_1_augmented_gn.csv` file). This was an heuristic dataset we used to further benchmark the proposed methodology.
  - The file `data_3.csv` is obtained from `data_2.csv` with SMOTE-data-extension (see the `data_1_augmented_gn.csv` file). This was an heuristic dataset we used to further benchmark the proposed methodology.
- `Plots` contains all the plots and figures of the project (in particular, those obtained via exploratory data analysis).
- `Processing` contains processing functions as the data augmentation scripts (e.g. `data_gn_adder.py` for gaussian-noise-data-extension), data aggregation (via `data_merger.py`) and the exploratory data analysis in èxploratory_data_analysis.py`.
- `Models` contains all the models we tested:
  - multi-layer-perceptrons (MLPs) both for single targets (e.g. `MLP_size.py`) or for joint targets (via `MLP_size_pdi.py`)
  - random forest regressors (both for single and joint targets)
  - extreme gradient boosting methods (both for single and joint targets)
  - variational autoencoders (a single try with the feature "size")
 - `requirements.txt` contains all the requirements needed in the project execution.

   
**Remark .** The best models have been obtained with the scripts `xgboost_one_target.py` with the results as in the logs `xgboost_one_target_size.log` and `xgboost_one_target_pdi.log` on `data_1.csv` with an extensive grid search, and cross-validation.


## Use this repository
If you want to use the code in this repository in your projects, please cite explicitely our work, and
* Clone the repository with `git clone https://github.com/leonardoLavagna/liposomes_ai`
* Install the requirements with `pip install -r requirements.txt`


## Contributing
We welcome contributions to enhance the functionality and performance of the models. Please submit pull requests or open issues for any improvements or bug fixes.

## License
This project is licensed under the MIT License.


## Citation
Cite this repository or one of the associated papers, such as:

```
....
```
