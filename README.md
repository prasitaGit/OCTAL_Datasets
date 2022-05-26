# OCTAL_Datasets

The specifications for Short is in: LTLset/ltlShort
The specifications for RERS is in: LTLset/ltlRERS
The specification for Diverse is in: LTLset/ltlDiverse

To generate the dataset, we need to run datasetTriPartiteModularize parameterized by the specification path and the corresponding system path:

As an example, the code below would generate the Diverse dataset. numneg represents the number of negative samples for every positive sample. By default it is 1,
i.e., one would get a 50-50 split

python datasetTriPartiteModularize.py --spec 'LTLset/ltlDiverse' --system 'BAset/BADiverse/*' --numneg 1

To run OCTAL with parameters, we would need to run main and set the variables root_path, data_train and data_test to the train and test datasets. Ex:

python main.py--root_path ‘homes/User/‘ --data_train_path ‘Diverse.pt' --data_test_path ‘RERS.pt’
