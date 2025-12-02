def check_true_label(data, mc):
    # put bread in blue plate
    obj_mc = mc.split(' in')[0].split('put ')[1]
    loc_mc = ' '.join(mc.split(' ')[-2:])
    if 'plate' not in loc_mc:
        breakpoint()

    # if object already moved to the target location, then false
    for obj_true, loc_true in data['action']:
        if obj_mc == obj_true and loc_mc == loc_true:
            # TODO: put lime in listed here???
            return 'False'

    #
    if obj_mc in data['init']['objects_target_like']:
        if loc_mc == data['init']['location_like']:
            true_label = 'True'
        else:
            true_label = 'False'
    elif obj_mc in data['init']['objects_target_dislike']:
        if loc_mc == data['init']['location_dislike']:
            true_label = 'True'
        else:
            true_label = 'False'
    else:
        true_label = 'False'
        # breakpoint()
        # raise ValueError(
        #     'obj_mc not in objects_target_like or objects_target_dislike'
        # )
    return true_label


def get_true_action(data):
    """Get a possible true action"""

    true_action = []
    for obj in data['init']['objects']:
        flag_possible = True
        for obj_true, loc_true in data['action']:
            if obj in data['init']['objects_target_like'
                                  ] and obj == obj_true and loc_true == data[
                                      'init']['location_like']:
                flag_possible = False  # already moved
                break
            if obj in data['init']['objects_target_dislike'
                                  ] and obj == obj_true and loc_true == data[
                                      'init']['location_dislike']:
                flag_possible = False  # already moved
                break

        if obj in data['init']['objects_target_like']:
            loc = data['init']['location_like']
        elif obj in data['init']['objects_target_dislike']:
            loc = data['init']['location_dislike']
        if flag_possible:
            true_action.append([obj, loc])

    # TODO: fix ambiguous one!!!
    return true_action