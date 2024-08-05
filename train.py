"""
   Copyright 2023 Universitat Politècnica de Catalunya

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import os

# 禁用GPU，只使用CPU进行运算
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import random
import time
from matplotlib import pyplot as plt
import numpy as np
from typing import List, Optional, Union, Tuple, Dict, Any

# Run eagerly-> Turn true for debugging only
# 设置TensorFlow函数是否急切执行，便于调试
RUN_EAGERLY = False
tf.config.run_functions_eagerly(RUN_EAGERLY)


def _reset_seeds(seed: int = 42) -> None:
    """Reset rng seeds, and also indicate tf if to run eagerly or not
        重置随机数生成器的种子，以确保结果可复现
    Parameters
    ----------
    seed : int, optional
        Seed for rngs, by default 42
    """
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def get_default_callbacks() -> List[tf.keras.callbacks.Callback]:
    """Returns the default callbacks for the training of the models
    获取模型训练的默认回调函数列表，包括提前停止和学习率衰减
    (EarlyStopping and ReduceLROnPlateau callbacks)
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            min_delta=0.0003,
            start_from_epoch=4,
        ),
        #监控的目标  等待的轮次（最多10epochs） 最小该变量  预热轮次
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=5,
            verbose=1,
            mode="min",
            min_delta=0.001,
        ),
        #改进率 等待轮次 1：更新消息  不再提高时减少改进率  门限
    ]


def get_default_hyperparams() -> Dict[str, Any]:
    """Returns the default hyperparameters for the training of the models. 
    返回模型训练的默认超参数设置
    包括 优化器、损失函数、额外指标、回调函数和训练周期数的字典
    - Adam optimizer with lr=0.001
    - MeanAbsolutePercentageError loss 自制损失函数
    - No additional metrics 
    - EarlyStopping and ReduceLROnPlateau callbacks
    - 100 epochs
    """
    return {
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
        "loss": tf.keras.losses.MeanAbsolutePercentageError(),#MAPE
        "metrics": ['MeanAbsolutePercentageError'],
        "additional_callbacks": get_default_callbacks(),
        "epochs": 100,
    }


def get_min_max_dict(
    ds: tf.data.Dataset, params: List[str], include_y: Optional[str] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Get the min and the max-min for different parameters of a dataset. Later used by the models for the min-max normalization.
    从数据集中获取参数的最小值和最大-最小值，用于归一化处理
    Parameters
    ----------
    ds : tf.data.Dataset
        Training dataset where to base the min-max normalization from.
        基于此数据集进行归一化的训练数据集
    params : List[str]
        List of strings indicating the parameters to extract the features from.
        要从中提取特征的参数列表
    include_y : Optional[str], optional
        Indicates if to also extract the features of the output variable.
        Inputs indicate the string key used on the return dict. If None, it is not included.
        By default None.
        是否同时提取输出变量的特征。如果提供,将使用此字符串键返回字典。默认为None。
    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary containing the values needed for the min-max normalization.
        The first value is the min value of the parameter, and the second is 1 / (max - min).
        包含归一化所需值的字典.第一个值是参数的最小值,第二个值是1/(最大值 - 最小值)
    """

    # Use first sample to get the shape of the tensors
    # 使用数据集的第一个样本获取张量的形状
    iter_ds = iter(ds)
    sample, label = next(iter_ds)
    params_lists = {param: sample[param].numpy() for param in params}
    #param: sample[param].numpy()：这是字典的键值对。param是字典的键，sample[param].numpy()是对应的值。
    if include_y:
        params_lists[include_y] = label.numpy()

    # Include the rest of the samples
    # 包含剩下的样本
    for sample, label in iter_ds:
        for param in params:
            params_lists[param] = np.concatenate(
                (params_lists[param], sample[param].numpy()), axis=0
            )
        if include_y:
            params_lists[include_y] = np.concatenate(
                (params_lists[include_y], label.numpy()), axis=0
            )

    scores = dict()
    for param, param_list in params_lists.items():
        min_val = np.min(param_list, axis=0)
        min_max_val = np.max(param_list, axis=0) - min_val
        if min_max_val.size == 1 and min_max_val == 0:
            scores[param] = [min_val, 0]
            print(f"Min-max normalization Warning: {param} has a max-min of 0.")
        elif min_max_val.size > 1 and np.any(min_max_val == 0):
            min_max_val[min_max_val != 0] = 1 / min_max_val[min_max_val != 0]
            scores[param] = [min_val, min_max_val]
            print(
                f"Min-max normalization Warning: Several values of {param} has a max-min of 0."
            )
        else:
            scores[param] = [min_val, 1 / min_max_val]

    return scores

def get_mean_std_dict(
    ds: tf.data.Dataset, params: List[str], include_y: Optional[str] = None
)-> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Get the mean and the std for different parameters of a dataset. Later used by the models for the standard normalization.
    从数据集中获取参数的平均值和标准差，用于标准化处理
    Parameters
    ----------
    ds : tf.data.Dataset

        Training dataset where to base the standard normalization from.
    
    params:List[str]

        List of strings indicating the parameters to extract the features from.

    include_y:Optional[str],optional

        Indicates if to also extract the features of the output variable.

    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary containing the values needed for the standard normalization.
        The first value is the mean value of the parameter, and the second is the std value of the parameter.
        包含标准正态化所需值的字典。第一个值是参数的平均值，第二个值是参数的标准差。
        
    """

    # Use first sample to get the shape of the tensors
    # 使用数据集的第一个样本获取张量的形状
    iter_ds = iter(ds)
    sample, label = next(iter_ds)
    params_lists = {param: sample[param].numpy() for param in params}
    #param: sample[param].numpy()：这是字典的键值对。param是字典的键，sample[param].numpy()是对应的值。
    if include_y:
        params_lists[include_y] = label.numpy()

    # Include the rest of the samples
    # 包含剩下的样本
    for sample, label in iter_ds:
        for param in params:
            params_lists[param] = np.concatenate(
                (params_lists[param], sample[param].numpy()), axis=0
            )
        if include_y:
            params_lists[include_y] = np.concatenate(
                (params_lists[include_y], label.numpy()), axis=0
    )
    scores=dict()

    for param, param_list in params_lists.items():
        mean_val = np.mean(param_list, axis=0)
        std_val = np.std(param_list, axis=0)
        
        if all(std_val) == 0:
            scores[param] = [mean_val, std_val]
        else:
            scores[param] = [mean_val, 1/std_val]
    return scores

 


def train_and_evaluate(
    ds_path: Union[str, Tuple[str, str]],
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss: tf.keras.losses.Loss,
    metrics: List[tf.keras.metrics.Metric],
    additional_callbacks: List[tf.keras.callbacks.Callback],
    dataset_type,
    epochs: int = 150,
    ckpt_path: Optional[str] = None,
    tensorboard_path: Optional[str] = None,
    restore_ckpt: bool = False,
    final_eval: bool = True,
) -> Tuple[tf.keras.Model, Union[float, np.ndarray, None]]:
    """
    Train the given model with the given dataset, using the provided parameters
    Besides for defining the hyperparameters, refer to get_default_hyperparams()

    Parameters
    ----------
    ds_path : str
        Path to the dataset. Datasets are expected to be in tf.data.Dataset format, and to be compressed with GZIP.
        If ds_path is a string, then it used as the path to both the training and validation dataset.
        If so, it is expected that the training and validation datasets are located in "{ds_path}/training" and "{ds_path}/validation" respectively.
        If ds_path is a tuple of two strings, then the first string is used as the path to the training dataset,
        and the second string is used as the path to the validation dataset.

    model : tf.keras.Model
        Instance of the model to train. Besides being a tf.keras.Model, it should have the same constructor and the name parameter
        as the models in the models module.

    optimizer : tf.keras.Optimizer
        Optimizer used by the training process

    loss : tf.keras.losses.Loss
        Loss function to be used by the process

    metrics : List[tf.keras.metrics.Metric]
        List of additional metrics to consider during training

    additional_callbacks : List[tf.keras.callbacks.Callback], optional
        List containing tensorflow callback functions to be added to the training process.
        A callback to generate tensorboard and checkpoint files at each epoch is already added.

    epochs : int, optional
        Number of epochs of in the training process, by default 100

    ckpt_path : Optional[str], optional
        Path where to store the training checkpints, by default "{repository root}/ckpt/{model name}"

    tensorboard_path : Optional[str], optional
        Path where to store tensorboard logs, by default "{repository root}/tensorboard/{model name}"

    restore_ckpt : bool, optional
        If True, before training the model, it is checked if there is a checkpoint file in the ckpt_path.
        If so, the model loads the latest checkpoint and continues training from there. By default False.

    final_eval : bool, optional
        If True, the model is evaluated on the validation dataset one last time after training, by default True

    Returns
    -------
    Tuple[tf.keras.Model, Union[float, np.ndarray, None]]
        Instance of the trained model, and the result of its evaluation
    """

    # Reset tf state #调用函数
    _reset_seeds()
    # Check epoch number is valid
    assert epochs > 0, "Epochs must be greater than 0"
    # Load ds
    if isinstance(ds_path, str):
        ds_train = tf.data.Dataset.load(f"{ds_path}/training", compression="GZIP")
        ds_val = tf.data.Dataset.load(f"{ds_path}/validation", compression="GZIP")
    else:
        ds_train = tf.data.Dataset.load(ds_path[0], compression="GZIP")
        ds_val = tf.data.Dataset.load(ds_path[1], compression="GZIP")
    # Checkpoint path # 如果没有指定检查点（checkpoint）路径，使用默认路径
    if ckpt_path is None:
        ckpt_path = f"ckpt/{model.name}"
    # Tensorboard path
    if tensorboard_path is None:
        tensorboard_path = f"tensorboard/{model.name}"

    # Apply min-max normalization 计算并设置数据归一化参数
    #model.set_min_max_scores(get_min_max_dict(ds_train, model.min_max_scores_fields))
    model.set_mean_std_scores(get_mean_std_dict(ds_train, model.mean_std_scores_fields))


    # Compile model 包括设置优化器、损失函数和其他度量标准。
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        run_eagerly=RUN_EAGERLY,
    )
    # Load checkpoint 根据 restore_ckpt 参数决定是否从检查点恢复模型 即接着之前的模型接着训练
    if restore_ckpt:
        ckpt = tf.train.latest_checkpoint(ckpt_path)
        if ckpt is not None:
            print("Restoring from checkpoint")
            model.load_weights(ckpt)
        else:
            print(
                f"WARNING: No checkpoint was found at '{ckpt_path}', training from scratch instead..."
            )
    else:
        print("restore_ckpt = False, training from scratch")

    # Create callbacks 设置模型检查点回调和 TensorBoard 日志回调，以监控训练过程并进行模型保存
    cpkt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(ckpt_path, "{epoch:02d}-{val_loss:.4f}"),
        verbose=1,
        mode="min",
        save_best_only=False,
        save_weights_only=True,
        save_freq="epoch",
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_path, histogram_freq=1
    )
    t0=time.time()

    # Train model 使用 fit 方法开始训练模型，指定训练数据集、验证数据集、训练周期和回调函数
    history=model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=[cpkt_callback, tensorboard_callback] + additional_callbacks,
        use_multiprocessing=True,
    )
    #使用fit返回的参数中MAPE的值

    print(model)
    
    if final_eval:
        plt.plot(history.history['mean_absolute_percentage_error'])
        plt.plot(history.history['val_mean_absolute_percentage_error'])
        plt.title(f'{dataset_type}model MAPE')
        plt.ylabel('MAPE')
        plt.xlabel('epoch')
        plt.legend(['train','val'],loc='upper right')
        plt.savefig(f'train_val_MAPE_{dataset_type}')
        
        return model, model.evaluate(ds_val)
    else:
        return model, None

'''
usage:
# Train the baseline model using the CBR+MB dataset
python train.py -ds CBR+MB

# Train the baseline model using the CBR+MB dataset through 5-fold cross validation
python train.py -ds CBR+MB -cfv

'''
if __name__ == "__main__":
    import argparse
    import models

    parser = argparse.ArgumentParser(
        description="Train a model for flow delay prediction"
    )
    parser.add_argument("-ds", type=str, help="Either 'CBR+MB' or 'MB'", required=True)
    parser.add_argument("--ckpt_path",type=str,required=True)
    parser.add_argument(
        "-cfv", action="store_true", help="Perform cross-fold validation"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds for cross-fold validation. Default is 5. Ignored if -cf is not set",
    )
    args = parser.parse_args()

    # Check the scenario
    if args.ds == "CBR+MB":
        ds_path = "data/data_cbr_mb_cv"
        model = models.Baseline_cbr_mb_2
    elif args.ds == "MB":
        ds_path = "data/data_mb_cv"
        model = models.Baseline_mb
    else:
        raise ValueError("Unrecognized dataset")

    # code for simple training/validation
    if not args.cfv:
        _reset_seeds()
        ckpt_path = f"ckpt/{args.ckpt_path}/"
        trained_model, evaluation = train_and_evaluate(
            os.path.join(ds_path, "0"),
            model(),
            **get_default_hyperparams(),
            ckpt_path=ckpt_path,
            dataset_type=args.ds,
        )
        print("Final evaluation:", evaluation)

    # code for cross-fold validation
    else:
        trained_models = []
        trained_models_val_loss = []
        ckpt_path = f"ckpt/{model.name}_cv/"
        tensorboard_path = f"tensorboard/{model.name}_cv/"

        # Execute each fold
        for fold_idx in range(args.n_folds):
            print("***** Fold", fold_idx, "*****")
            _reset_seeds()
            trained_model, evaluation = train_and_evaluate(
                os.path.join(ds_path, str(fold_idx)),
                model(),
                **get_default_hyperparams(),
                dataset_type=args.ds,
                ckpt_path=os.path.join(ckpt_path, str(fold_idx)),
                tensorboard_path=os.path.join(tensorboard_path, str(fold_idx)),
            )
            trained_models.append(trained_model)
            trained_models_val_loss.append(evaluation)

        # Print final evaluation
        for fold_idx, evaluation in enumerate(trained_models_val_loss):
            print(f"Fold {fold_idx} evaluation:", trained_models_val_loss[fold_idx])
