import pandas as pd
import numpy as np
import cv2


DATA_PATH = '../data/'


def remove_zero_bias(df):
    rows_with_steering_zero = df[(df.steering == 0)]
    total_rows_with_steering_zero = len(rows_with_steering_zero)
    drop_indices = np.random.choice(rows_with_steering_zero.index, int(total_rows_with_steering_zero * 0.90),
                                    replace=False)

    return df.drop(drop_indices)


def augment_steering_angles_and_reshape_dataframe(df):
    rows = []

    for _, row in df.iterrows():
        rows.append({'image': row.left, 'steering': row.steering + .25, 'is_flipped': False})
        rows.append({'image': row.center, 'steering': row.steering, 'is_flipped': False})
        rows.append({'image': row.right, 'steering': row.steering - .25, 'is_flipped': False})

    return pd.DataFrame(rows)


def augment_with_horizontal_flip(df):
    df_flipped = df.copy()
    df_flipped.steering = df_flipped.steering.apply(lambda x: x * -1)
    df_flipped.is_flipped = True

    return pd.concat([df, df_flipped])


def augment_dataframe(df):
    df = remove_zero_bias(df)
    df = augment_steering_angles_and_reshape_dataframe(df)
    df = augment_with_horizontal_flip(df)

    return df


def crop_image(image):
    return image[32:132]


def load_image(image_path, is_flipped):
    image = cv2.imread(image_path)
    image = crop_image(image)

    if is_flipped:
        image = cv2.flip(image, 1)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)