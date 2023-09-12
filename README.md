# Variable_Importance
* The required packages are installed as a new conda environment including both R and Python dependencies with the following command:

```
conda env create -f requirements_conda.yml
```

* The missing R packages can be found in the "requirements_r.rda" file and can be downloaded using the following commands:

```
load("requirements_r.rda")

for (count in 1:length(installedpackages)) {
    install.packages(installedpackages[count])
}
```

* The ```permimp``` package can be downloaded with the following commands:

```
library(devtools)

install_github("Anonymous1346/permimp")
```

* For the 3 first experiments, ```compute_simulations``` is used along with ```plot_simulations_all```:
  * For the **first experiment**:
    * Uncomment both ```dnn_py``` and ```dnn_py_cond```
    * ```n_samples``` is set to 300 and ```n_featues``` is set to 100
    * Uncomment all the ```rho``` values
    * Set ```prob_sim_data``` to ```regression_perm```
    * The csv files ```AUC_blocks_100_Mi_dnn_dnn_py_300:100```, ```power_blocks_100_Mi_dnn_dnn_py_300:100```, ```type1error_blocks_100_Mi_dnn_dnn_py_300:100``` and ```time_bars_blocks_100_Mi_dnn_dnn_py_300:100``` are found in ```results/results_csv```
  
  * For the **second experiment**:
    * Keep both ```dnn_py``` and ```dnn_py_cond``` uncommented
    * Set ```n_samples``` to ```n_samples = `if`(!DEBUG, seq(100, 1000, by = 100), 10L)``` (Uncomment the line directly below)
    * Set ```n_features``` to 50
    * In ```prob_sim_data```, comment ```regression_perm``` and uncomment all the rest except ```regression_group_sim_1```
    * The csv files ```AUC_blocks_100_dnn_dnn_py_perm_100--1000``` and ```type1error_blocks_100_dnn_dnn_py_perm_100--1000``` are found in ```results/results_csv```

  * For the **third experiment**:
    * Uncomment all the methods
    * Set ```n_samples``` to 1000 and ```n_features``` to 50
    * In ```prob_sim_data```, comment ```regression_perm``` and ```
    * The csv files ```AUC_blocks_100_Mi_dnn_dnn_py_300:100```, ```power_blocks_100_Mi_dnn_dnn_py_300:100```, ```type1error_blocks_100_Mi_dnn_dnn_py_300:100``` and ```time_bars_blocks_100_Mi_dnn_dnn_py_300:100``` are found in ```results/results_csv```

  * Once the simulated data are computed, we move to the ```plot_simulations_all``` (Don't forget to change the name of the file to save with each experiment as it will be used later for the plots):
    * For the **first experiment**:
      * Change ```source``` to ```source("utils/plot_methods_all_Mi.R")```
      * Set ```run_plot_auc```, ```run_plot_type1error```, ```run_plot_power``` and ```run_time``` one by one to TRUE.
    
    * For the **second experiment**:
      * Change ```source``` to ```source("utils/plot_methods_all_increasing_combine.R")```.
      * Set ```run_plot_combine``` to TRUE.
      * Don't forget to change the name of the input and output files.
    
    * For the **third experiment**:
      * Change ```source``` to ```source("utils/plot_methods_all.R")```.
      * Set ```run_plot_auc```, ```run_plot_type1error```, ```run_plot_power``` and ```run_time``` one by one to TRUE.

  * Once the files are ready, we move to the ```visualization``` folder where the two notebooks ```plot_figure_simulations``` and ```plot_figure_simulations_2``` are used to plot the figures of the main paper and the supplementary material respectively.

  * For the **forth experiment**, we move to the ```ukbb``` folder:
    * The data are the public data from UKBB that needs to sign an agreement before using it (Any personal data are already removed)
    * In the ```process``` scripts, change method to ```permfit_dnn``` or ```cpi_dnn``` to process the data and explore the importance of the variables using one of the methods.
    * The plots of experiment 4 can be obtained by accessing the notebook ```plot_ukbb_results``` in the ```visualization``` folder.
