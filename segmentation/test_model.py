import os
from skimage import io
import sys
sys.path.append("../images_loaders/")
sys.path.append("../analyzing_datasets/")
import analyze_image_pair as A
import matplotlib.pyplot as plt
import Jaccard as J
from SD3 import sd3

def test_model(apply_model_function, provider, save_folder_path = None, printing_function = print):
    printing_function(f"Will try: {len(provider.img_files)} benchmark files.")

    accuracyHist = J.JaccardsHistogram()
    accuracyVals = { "avgJ":dict(), "cov":dict(), "sd3":dict() }
    accuracySD3s = []

    for i,m in iter(provider):
        res = apply_model_function(i)

        printing_function(f"going to compare ref {m.shape} with res {res.shape}")
        _, label_accuracies = J.Jaccard(res,m)
        accuracyHist.add_dict(label_accuracies)
        accuracyAvg, coverage = J.Jaccard_average(label_accuracies)
        printing_function(f"-> avg accuracy = {accuracyAvg:0.06f} with coverage = {coverage*100:2.02f}%")

        sd_value = sd3(res,m)
        printing_function(f"-> sd3 accuracy = {sd_value:0.06f}")
        accuracySD3s.append(sd_value)

        if save_folder_path is not None:
            path = os.path.join(save_folder_path, "SegImg_"+provider.last_used_img_filename())
            io.imsave(path, res, check_contrast=False)

            path = os.path.join(save_folder_path, "JaccImg_"+provider.last_used_img_filename())
            path = path[:path.rfind('.')]+".tif" # change suffix to .tif
            io.imsave(path, J.project_eval_into(m, label_accuracies), check_contrast=False)

            path = os.path.join(save_folder_path, "JaccHist_"+provider.last_used_img_filename())
            path = path[:path.rfind('.')]+".png" # change suffix to .png
            A.create_plot( accuracyHist.create_hist(label_accuracies) ) # histogram from label_accuracies
            A.save_plot_as_png(path)

            accuracyVals["avgJ"][provider.last_used_img_filename()] = accuracyAvg
            accuracyVals[ "cov"][provider.last_used_img_filename()] = coverage
            accuracyVals[ "sd3"][provider.last_used_img_filename()] = sd_value

    if save_folder_path is not None:
        figure_width = len(accuracyVals['avgJ']) +1
        # plot accuracyVals
        plt.close()
        plt.figure(figsize=(figure_width,5))
        plt.xticks(rotation=45, ha="right")
        plt.gca().set_ylim([-0.1,1.1])
        plt.plot(accuracyVals["avgJ"].keys(), accuracyVals["avgJ"].values(), "x", accuracyVals["avgJ"].keys(), accuracyVals["cov"].values(), "r_")
        plt.tight_layout()
        plt.savefig( os.path.join(save_folder_path, "Jaccards_per_file.png") )

        plt.close()
        plt.figure(figsize=(figure_width,5))
        plt.xticks(rotation=45, ha="right")
        #plt.gca().set_ylim([-0.1,1.1])
        plt.plot(accuracyVals["sd3"].keys(), accuracyVals["sd3"].values(), "x")
        plt.tight_layout()
        plt.savefig( os.path.join(save_folder_path, "SD3s_per_file.png") )

    avg_J,avg_cov = accuracyHist.report_average_Jaccard()
    printing_function(f"Average Jaccard = {avg_J:0.06f} collected over {accuracyHist.jacc_cnt} masks, but only {avg_cov*100:2.02f}% masks were discovered.")
    printing_function(f"Average SD3 = {sum(accuracySD3s)/len(accuracySD3s):0.06f}")

    return accuracyHist

