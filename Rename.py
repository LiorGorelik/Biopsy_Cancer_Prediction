import os

path = os.chdir("/home/lior.go@staff.technion.ac.il/PycharmProjects/pythonProject/nki_vgh_data/train/image")

for file in os.listdir(path):
    new_file_name = file.replace("0.1","1").replace("0.2","2").replace("0.3","3").replace("0.4","4")

    os.rename(file, new_file_name)
