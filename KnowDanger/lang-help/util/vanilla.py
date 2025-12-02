def check_true_label(ground_truth, mc, ambiguity_name):

    if 'left' in ground_truth:
        spatial = 'left'
    elif 'right' in ground_truth:
        spatial = 'right'
    elif 'front' in ground_truth:
        spatial = 'front'
    elif 'behind' in ground_truth:
        spatial = 'back'
    else:
        spatial = 'on'
    # first get move objects
    if ambiguity_name == 'numeric':
        num_obj = ground_truth.split(' ')[1]
        if num_obj == 'one':
            num_obj = 1
        elif num_obj == 'two':
            num_obj = 2
        elif num_obj == 'three':
            num_obj = 3
        move_obj_type = ground_truth.split(' ')[2]
        if move_obj_type[-1] == 's':
            move_obj_type = move_obj_type[:-1]
    else:
        num_obj = 1
        # put the blue block at the right side of the green bowl
        move_obj = ' '.join(ground_truth.split(' ')[2:4])
    # get target object
    target_obj = ' '.join(ground_truth.split(' ')[-2:])
    if target_obj[-1] == 's':
        target_obj = target_obj[:-1]

    # get spatial and target obj in mc
    move_obj_mc = ' '.join(mc.split(' ')[1:3])  # move green block
    if 'left' in mc:
        spatial_mc = 'left'
    elif 'right' in mc:
        spatial_mc = 'right'
    elif 'front' in mc:
        spatial_mc = 'front'
    elif 'back' in mc or 'behind' in mc:
        spatial_mc = 'back'
    else:
        spatial_mc = 'on'
    target_obj_mc = ' '.join(mc.split(' ')[-2:])

    if ambiguity_name == 'numeric':
        # move green block and yellow block and blue block to add_left_offset_from_obj_pos('green bowl')
        move_obj_type_mc = mc.split(' ')[
            2]  # assume same type for all objects to be moved
        num_obj_mc = mc.count('and') + 1

        # check if all move obj the same type
        true_label = 'True'
        move_obj_all = mc.split('put')[1].split('and')
        for move_obj in move_obj_all:
            if 'to' in move_obj:
                move_obj = move_obj.split('to')[0].strip()
            elif 'on' in move_obj:
                move_obj = move_obj.split('on')[0].strip()
            if move_obj_type_mc not in move_obj:
                true_label = 'False'
            break
        if true_label == 'False':
            return true_label

        if num_obj == num_obj_mc and move_obj_type == move_obj_type_mc and target_obj_mc == target_obj and spatial == spatial_mc:
            true_label = 'True'
        else:
            true_label = 'False'
    else:
        if spatial == spatial_mc and move_obj == move_obj_mc and target_obj == target_obj_mc:
            true_label = 'True'
        else:
            true_label = 'False'
    return true_label
