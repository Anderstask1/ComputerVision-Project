def cell_phone_detection(text, phone_count):
    # if phone is detected, count number of frames
    if text.split(":")[0] == "cell phone":
        return phone_count + 1
    else:
        return 0