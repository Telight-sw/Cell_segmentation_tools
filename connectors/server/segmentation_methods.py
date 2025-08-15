import os
import sys
sys.path.append("../../SLURM/testing_evaluation")
import common_tools

sys.path.append("../../segmentation/cellpose")
sys.path.append("../../segmentation/instanseg")
sys.path.append("../../segmentation/MaskRCNN")
import cellpose_wrapper as C
import instanseg_wrapper as I
import maskrcnn_wrapper as M


class SegmentationMethods:
    def __init__(self, methods_folder: str = '.'):
        # a map between method name and its file
        self.available_methods = {}
        self.rescan_methods(methods_folder)

        self.last_used_method = "__intentionally_invalid_method_name__"
        self.last_used_fun = None


    def rescan_methods(self, methods_folder: str = '.'):
        '''
        Search for *.*.networkName.model files in the 'methods_folder',
        and return them as a map (dict) between method name and its file.
        '''

        methods = common_tools.list_models_files(methods_folder)
        self.available_methods = {}

        for m in methods:
            ds,net,t,bs,e = common_tools.folder_name_to_atoms(m)
            self.available_methods[f"{net}.{ds}"] = os.path.join(methods_folder, m)

        print("Discovery of available networks is finished.")


    def list_avail_methods(self):
        return self.available_methods.keys()


    def get_segmentation_fun(self, wanted_method:str):
        # re-use the currently activated method
        if wanted_method == self.last_used_method:
            return self.last_used_fun

        # sanity check
        if wanted_method not in self.available_methods.keys():
            print("REQUESTED MODEL NOT AVAILABLE!")
            return None

        wanted_file = self.available_methods[wanted_method]
        wanted_net = wanted_method.split('.')[0]

        if wanted_net == "cellpose":
            print(f"LOADING CELLPOSE model: {wanted_file}")
            model = C.load_model(wanted_file)
            self.last_used_fun = lambda i : C.apply_model(model,i)
            self.last_used_method = wanted_method
            # NB: keep the 'last_used_method' only if switching to this method went well,
            #     otherwise by not memorizing it, the request to the same method will trigger
            #     _again_ this code path

        elif wanted_net == "maskrcnnv1COCO":
            print(f"LOADING MASKRCNN_v1COCO  model: {wanted_file}")
            model = M.create_official_model()
            M.load_model(model, wanted_file)
            self.last_used_fun = lambda i : M.apply_model(model,i)[0]
            self.last_used_method = wanted_method

        elif wanted_net == "instanseg":
            print(f"LOADING INSTANSEG model: {wanted_file}")
            model = I.load_model(wanted_file, subfolder='.', model_name_suffix='')
            self.last_used_fun = lambda i : I.apply_model(model,i)
            self.last_used_method = wanted_method

        else:
            print("NOT LOADING ANY MODEL!?")
            return None ## this should never happen

        return self.last_used_fun

