import logging


def setup_logging(root, path):
    """
    Set up the logging configuration.
    
    Args:
        root (str): The root directory where the log file will be stored.
        path (str): The relative path or filename for the log file.

    Returns:
        None
    """
    logging.basicConfig(
        filename=root + "/" + path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


logs_path = "_Logs"
data_path = "Data"
files_path = "_Files"
plots_path = "Plots"
raw_data_path = "_Raw_Data"
models_path = "_Models"


column_names = ["ID","ML","CHIP","ESM","HSPC",
                "CHOL","PEG","TFR","FRR","AQUEOUS",
                "OUTPUT","SIZE","PDI"]