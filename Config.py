


Initial_params = dict(DataBase_directory = "E:\Bazy_Danych\MNIST_Data",
                      Kaggle_set = True,
                      Load_from_CSV = True,
                      Stratification_test = False,
                      grayscale = True,
                      img_H = 28,
                      img_W = 28,
                      DataType = "float32"
                      )

#If train_dataset_multiplier set to one then there is no augmentation
Augment_params = dict(reduced_set_size = None,
                      dataset_multiplier = 3,
                      flipRotate = False,
                      randBright = True,
                      gaussian_noise = True,
                      denoise = False,
                      contour = False
                      )


Model_parameters = dict(model_architecture = "SimpleMnist",
                        device = "GPU:0",
                        train = True,
                        epochs = 30,
                        patience = 10,
                        batch_size = 128,
                        min_delta = 0.0001,
                        evaluate = True,
                        show_architecture = False
                       )


