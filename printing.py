def print2(*objects):
    # Print to default sys.stdout
    print(*objects)
    # Print to '.screen file'
    screen_file = open(fileOperations.screen_file_path,'a')
    objects_esc = list()
    for i in range(len(objects)):
        objects_esc.append(escapeANSI(objects[i]))
    print(*objects_esc,file = screen_file)
    screen_file.close()
