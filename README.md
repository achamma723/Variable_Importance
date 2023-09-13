# Variable_Importance
* The required packages are installed as a new conda environment including both R and Python dependencies with the following command:

```
conda create --name envbase --file requirements_conda.yml
```

* The ```missing R packages``` can be found in the "requirements_r.rda" file and can be downloaded using the following commands:

```
load("requirements_r.rda")

for (count in 1:length(installed_packages)) {
    install.packages(installed_packages[count])
}
```

> :warning: For ```reticulate```, if asked for default python virtual environment, the answer should be ```no``` to take the default *conda* environment into consideration

* The ```permimp``` package can be downloaded with the following commands:

```
install.packages('permimp', repos=NULL, type='source')
```

* The ```sandbox``` package can be downloaded going inside "code_dcrt" with:

```
* python setup.py build_ext --inplace
* pip install -e 
```

# Computation
## Computation files
### Main text
* For the 3 first experiments, ```compute_simulations``` is used along with
  ```plot_simulations_all```:
  * Set ```N_SIMULATIONS``` to *1:100* to perform the 100 runs.
  * Set ```N_CPU``` according to the reserved resources (*parallel*) or 1 (*serial*).
  * For the **first experiment**:
    * Set ```DEBUG``` to FALSE.
    * Uncomment both ```permfit``` and ```cpi```.
    * ```n_samples``` is set to 300 and ```n_featues``` is set to 100
    * Uncomment all the ```rho``` values.
    * Set ```prob_sim_data``` to ```regression_perm```.
    * In ```stat_knockoff```, uncomment (```lasso_cv```).
    * The *output csv* file
      ```simulation_results_blocks_100_Mi_dnn_dnn_py_300:100``` is found in ```results/results_csv```.
  
  * For the **second experiment**:
    * Set ```DEBUG``` to FALSE.
    * Uncomment both ```permfit``` and ```cpi```.
    * Set ```n_samples``` to ```n_samples = `if`(!DEBUG, seq(100, 1000, by =
      100), 10L)``` (comment *line 84* and uncomment *line 85*).
    * Set ```n_features``` to 50
    * In ```prob_sim_data```, comment ```regression_perm``` and uncomment all
      the rest.
    * In ```stat_knockoff```, uncomment (```lasso_cv```).
    * The *output csv* file
      ```simulation_results_blocks_100_dnn_dnn_py_perm_100--1000``` is found in ```results/results_csv```.

  * For the **third experiment**:
    * Uncomment all methods.
    * Set ```n_samples``` to *1000* and ```n_features``` to *50*.
    * In ```prob_sim_data```, comment ```regression_perm``` and uncomment all
      the rest.
    * In ```stat_knockoff```, uncomment (```lasso_cv```, ```bart``` and ```deep```).
    * The *output csv* file
      ```simulation_results_blocks_100_allMethods_pred_final``` is found in ```results/results_csv```.

  * For the **forth experiment**, we move to the ```ukbb``` folder:
    * The data are the public data from UK Biobank that needs to sign an agreement before using it (Any personal data are already removed).
    * In the ```process``` scripts, change method to ```permfit_dnn``` or
      ```cpi_dnn``` to process the data and explore the importance of the
      variables using one of the methods.
    * The corresponding results per method are found in ```Results_variables``` folder.
  
### Supplementary Material
  * For the **section D**:
    * Set ```DEBUG``` to FALSE.
    * Uncomment both ```cpi``` and ```loco_dnn``` (*The last item uncommitted
      shouldn't be followed by a comma*).
    * Set ```n_samples``` to *1000*, ```n_features``` to *50* and ```rho``` to
      *0.8*.
    * In ```prob_sim_data```, uncomment ```regression```.
  * The *output csv* file
    ```simulation_results_blocks_100_CPI_LOCO_DNN``` is found in ```results/results_csv```.

  * For the **section M**:
    * We use ``` compute_simulations_py```.
    * Large scale simulation:
      * The script can be launched with the following command:
        ```
        python -u compute_simulations_py.py --n 10000 --p 50 --nsig 20 --nblocks 10 --intra 0.8 --conditional 1 --f 1 --s 100 --njobs 1
        ```
        * ```--n``` stands for the number of samples
        * ```--p``` stands for the number of variables
        * ```--nsig``` stands for the number of significant variables randomly chosen
        * ```--nblocks``` stands for the number of blocks/groups in the data
          structure
        * ```--intra``` stands for the intra correlation inside the groups
        * ```--conditional``` stands for the use of CPI (`1`) or PI (`0`)
        * ```--f``` stands for the first point of the range (Default `1`)
        * ```--s``` stands for the step-size i.e. range size (Default `100`)
        * ```--njobs``` stands for the serial/parallel implementation under
          `Joblib` (Default `1`)
        * The *csv output* file
          ```simulation_results_blocks_100_n_10000_p_50_cpi_permfit``` is found
          in ```results/results_csv```.

    * UK Biobank semi-simulation:
      * The ```filename``` should be changed to the corresponding UKBB data
        (not publicly available).
      * The script can be launched with the following command:
        ```
        python -u compute_simulations_py.py --nsig 115 --conditional 1 --f 1 --s 100 --njobs 1
        python -u compute_simulations_py.py --nsig 115 --conditional 0 --f 1 --s 100 --njobs 1
        ```
        * ```--nsig``` stands for the number of significant variables randomly chosen
        * ```--conditional``` stands for the use of CPI (`1`) or PI (`0`)
        * ```--f``` stands for the first point of the range (Default `1`)
        * ```--s``` stands for the step-size i.e. range size (Default `100`)
        * ```--njobs``` stands for the serial/parallel implementation under
          `Joblib` (Default `1`)
        * The *csv output* file
          ```simulation_results_blocks_100_UKBB_single``` is found
          in ```results/results_csv```.

  * For the **section N**:
    * The Cam-CAN data is not publicly available, thus we provide the script
      *process_age_prediction_CamCAN* in order to compute the degree of
      significance for each frequency band.
    * The *output csv* file ```Result_single_FREQ_all_imp_outer_10_inner``` is
      found in ```camcan```.

## Plotting files
  * We move to the ```plot_simulations_all```:
### Main text
  * For the **first experiment** with
    *simulation_results_blocks_100_Mi_dnn_dnn_py_300:100* as input:
    * Change ```source``` (*at line 2*) to
      ```source("utils/plot_methods_all_Mi.R")```.
    * Set ```nb_relevant``` to *20* and ```N_CPU``` to the number of dedicated resources.
    * Set ```run_plot_auc```, ```run_plot_type1error```, ```run_plot_power```
      and ```run_time``` one by one to TRUE.
    * Set ```run_plot_combine``` and
      ```run_all_methods``` to FALSE.
    * Uncomment (```Permfit-DNN``` and ```CPI-DNN```).
    * The *output csv* files ```AUC_blocks_100_Mi_dnn_dnn_py_300:100```, ```power_blocks_100_Mi_dnn_dnn_py_300:100```, ```type1error_blocks_100_Mi_dnn_dnn_py_300:100``` and ```time_bars_blocks_100_Mi_dnn_dnn_py_300:100``` are found in ```results/results_csv```.
  
  * For the **second experiment** with
    *simulation_results_blocks_100_dnn_dnn_py_perm_100--1000* as input:
    * Change ```source``` (*at line 2*) to
      ```source("utils/plot_methods_all_increasing_combine.R")```.
    * Set ```nb_relevant``` to *20* and ```N_CPU``` to the number of dedicated resources.
    * Set ```run_plot_combine``` to TRUE.
    * Set ```run_all_methods``` to FALSE.
    * Set ```run_plot_auc```, ```run_plot_type1error```, ```run_plot_power```
      and ```run_time``` one by one to FALSE.
    * Uncomment (```Permfit-DNN``` and ```CPI-DNN```).
    * The *output csv* files ```AUC_blocks_100_dnn_dnn_py_perm_100--1000``` and
      ```type1error_blocks_100_dnn_dnn_py_perm_100--1000``` are found in
      ```results/results_csv```.

  * For the **third experiment** with
    *simulation_results_blocks_100_allMethods_pred_final* as input:
    * Change ```source``` (*at line 2*) to
      ```source("utils/plot_methods_all.R")```.
    * Set ```nb_relevant``` to *20* and ```N_CPU``` to the number of dedicated resources.
    * Set ```run_plot_auc```, ```run_plot_type1error```, ```run_plot_power``` one by one to TRUE.
    * Set ```run_plot_combine``` and ```run_time``` to FALSE.
    * Set ```run_all_methods``` and ```with_pval``` to TRUE.
    * Uncomment (```Marg```, ```d0CRT```, ```Permfit-DNN```, ```CPI-DNN```,
      ```CPI-RF```, ```lazyvi```, ```cpi_knockoff```, ```loco``` and ```Strobl```).
    * The *output csv* files
      ```AUC_blocks_100_allMethods_pred_imp_final_withPval```,
      ```power_blocks_100_allMethods_pred_imp_final``
      and ```type1error_blocks_100_allMethods_pred_imp_final``` are found in
      ```results/results_csv```.

### Supplementary Material
  * For the supplementary experiments:
    * For the **section D** with *simulation_results_blocks_100_CPI_LOCO_DNN* as
      input:
      * Change ```source``` (*at line 2*) to
        ```source("utils/plot_methods_all.R")```.
      * Set ```nb_relevant``` to *20* and ```N_CPU``` to the number of dedicated resources.
      * Set ```run_plot_auc```, ```run_plot_type1error```, ```run_plot_power```
        and ```run_time``` one by one to TRUE.
      * Set ```run_plot_combine``` and
        ```run_all_methods``` to FALSE.
      * Uncomment (```LOCO-DNN``` and ```CPI-DNN```).
      * The *output csv* files ```AUC_blocks_100_CPI_LOCO_DNN```,
        ```power_blocks_100_CPI_LOCO_DNN```,
        ```type1error_blocks_100_CPI_LOCO_DNN``` and
        ```time_bars_blocks_100_CPI_LOCO_DNN``` are found in
        ```results/results_csv```.

    * For the **section I** with *simulation_results_blocks_100_allMethods_pred_final* as input:
      * Change ```source``` (*at line 2*) to
        ```source("utils/plot_methods_all.R")```.
      * Set ```nb_relevant``` to *20* and ```N_CPU``` to the number of dedicated resources.
      * Set ```run_plot_auc``` to TRUE.
      * Set ```run_plot_type1error```, ```run_plot_power```, ```run_time``````run_plot_combine``` and ```with_pval``` to FALSE.
      * Set ```run_all_methods``` to TRUE.
      * Uncomment (```Knockoff_bart```, ```Knockoff_lasso```, ```Shap```, ```SAGE```,
        ```MDI```, ```BART```, ```Knockoff_deep```, ```Knockoff_path``` and ```Knockoff_lasso```).
      * The *output csv* file
        ```AUC_blocks_100_allMethods_pred_imp_final_withoutPval``` is found in
        ```results/results_csv```.

    * For the **section K** with *simulation_results_blocks_100_allMethods_pred_final* as input:
      * Change ```source``` (*at line 2*) to
        ```source("utils/plot_methods_all.R")```.
      * Set ```run_time``` to TRUE and the rest to FALSE.
      * Uncomment all the methods.
      *  The *output csv* file
         ```time_bars_blocks_100_allMethods_pred_imp_final``` is found in
        ```results/results_csv```.

    * For the **section M**:
      * Change ```source``` (*at line 2*) to
        ```source("utils/plot_methods_all.R")```.
      * Set ```run_plot_auc```, ```run_plot_type1error```, ```run_plot_power```
        and ```run_time``` one by one to TRUE.
      * Set ```run_plot_combine```, ```run_all_methods``` and ```with_pval``` to
        FALSE.
      * Uncomment (```Permfit-DNN```, ```CPI-DNN```).
      * Large scale simulation with *simulation_results_blocks_100_n_10000_p_50_cpi_permfit* as input:
        * Set ```nb_relevant``` to *20* and ```N_CPU``` to the number of dedicated resources.
        * The *output csv* files are found in ```results/results_csv``` under
          ```[AUC-type1error-power-time_bars]_blocks_100_groups_CPI_n_10000_p_50_cpi_permfit```.
      * UK Biobank semi simulation:
        * Set ```nb_relevant``` to *115* and ```N_CPU``` to the number of
          dedicated resources.
        * The *output csv* files are found in ```results/results_csv``` under
          ```[AUC-type1error-power-time_bars]_blocks_100_UKBB_single```.

# Plotting part

* We move to the ```visualization``` with *4* notebooks
  ```plot_figure_simulations```, ```plot_figure_simulations_2```, ```plot_figure_simulations_3```
  ```plot_ukbb_results``` and ```plot_freqRes```:
    * ```plot_figure_simulations``` for the plots in the *main text*.
    * ```plot_figure_simulations_2``` and ```plot_figure_simulations_3``` for
      the plots in the *supplement*.
    * ```plot_ukbb_results``` for the plot of the *forth experiment*.
    * ```plot_freqRes``` for the Cam-CAN corresponding plot.

