import os
import json
from .data_utils.data_loader import lsp_load_data
from .loss import l2_loss
from .evaluation import accu_rate

def find_latest_checkpoint(checkpoints_path):
    ep = 0
    r = None
    while True:
        if os.path.isfile(checkpoints_path + "." + str(ep)):
            r = checkpoints_path + "." + str(ep)
        else:
            return r

        ep += 1


def train(model,
          train_images_path,
          train_fnames_txt_path,
          jts_map,
          resize_height=None,  # user-specified. It will affects the cropping.
          resize_width=None,  # user-specified. It will affects the cropping.
          n_classes=None,  # pre-defined by JtNet
          checkpoints_path=None,
          epochs=5,
          batch_size=2,
          validate=False,
          val_images_path=None,
          val_fnames_txt_path=None,
          val_batch_size=2,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=800,
          steps_val_per_epoch=100,
          optimizer_name='adadelta'
          ):

    train_history_list = []

    model.compile(loss=l2_loss,
                  optimizer=optimizer_name,
                  metrics=[accu_rate])

    if not checkpoints_path is None:
        open(checkpoints_path + "_config.json", "w").write(json.dumps({
            "n_classes": n_classes,
            "input_height": resize_height,
            "input_width": resize_width,
            # "output_height" : output_height ,
            # "output_width" : output_width
        }))

    if (not (load_weights is None)) and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (not checkpoints_path is None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if not latest_checkpoint is None:
            print("Loading the weights from latest checkpoint ", latest_checkpoint)
            model.load_weights(latest_checkpoint)

    train_gen = lsp_load_data(train_images_path,
                              train_fnames_txt_path,
                              jts_map,
                              batch_size=batch_size,
                              n_classes=n_classes,
                              person_centered_cropped=True,
                              resize_height=resize_height,
                              resize_width=resize_width,
                              return_originals=False
                              )

    if validate:
        val_gen = lsp_load_data(val_images_path,
                                val_fnames_txt_path,
                                jts_map,
                                batch_size=val_batch_size,
                                n_classes=n_classes,
                                person_centered_cropped=True,
                                resize_height=resize_height,
                                resize_width=resize_width,
                                return_originals=False
                                )
        for ep in range(epochs):
            print("Starting Epoch ", ep)

            H = model.fit_generator(train_gen, steps_per_epoch, validation_data=val_gen,
                                    validation_steps=steps_val_per_epoch, epochs=1)
            train_history_list.append(H.history)

            print("attempting ep:", ep)
            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))
            print("Finished Epoch", ep)

    else:
        for ep in range(epochs):
            print("Starting Epoch ", ep)

            H = model.fit_generator(train_gen, steps_per_epoch, epochs=1)
            train_history_list.append(H.history)

            if not checkpoints_path is None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))
            print("Finished Epoch", ep)


    return train_history_list