from detect_cell import detect

files = ['cell_datatset/images/train/Ascaris lumbricoides_0006.jpg','cell_datatset/images/train/Trichuris trichiura_0992.jpg']
images = detect(files)

print(images.shape)