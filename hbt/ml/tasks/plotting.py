import luigi
import law
import numpy as np

from hbt.tasks.base import HBTTask
from columnflow.tasks.framework.plotting import PlotBase 
import hbt.ml.plotting as plt


class BaseClass(law.Task):
    skip_uncertenties = luigi.parameter.BoolParameter()
    normalization = luigi.ChoiceParameter(var_type=str,
                          choices = ['row', 'col', 'None'])



    def local_target(self, file_name, **kwargs):
        return law.LocalFileTarget(
            "/nfs/dust/cms/user/haddadan/law_luigi/data/" + file_name, **kwargs
        )
    
    def requires(self):
        return {
            'true_labels':None, 
            'model_output':None, 
            'sample_weights':None, 
        }
    

class BasePlottingClass(BaseClass):
    file_typ = luigi.ChoiceParameter(var_type=str,
                                     choices = ['png', 'pdf', 'ps', 'eps' , 'svg'])
    process_labels=luigi.ListParameter(None)
    class_labels=luigi.ListParameter(None)
    plot_title=luigi.OptionalStrParameter(default='')
    colormap=luigi.ChoiceParameter(var_type=str,
                                   choices=['cf_green_cmap','cf_ygb_cmap','cf_cmap','viridis'])
    digits=luigi.IntParameter(default=3)

class CreateConfusionMatrix(BaseClass):

    def output(self):
        return self.local_target(
            'cm_data_{}.npz'.format('something')
        ) 
    
    def run(self):
        cm = plt.get_conf_matrix(true_labels=self.input(), 
                        model_output=self.input(), 
                        sample_weights=self.input(), 
                        normalization=self.normalization, 
                        errors=not(self.skip_uncertenties))


class PlotConfusionMatrix(BasePlottingClass):

    def requires(self):
        return CreateConfusionMatrix.req(self)
    
    def output(self):
        return self.local_target(
            'cm_{}.{}'.format('something', self.file_typ)
        )
    
    def run(self):
        cm = np.load(self.input().abspath)
        z_title = 'Accuracy'
        if self.normalization != 'None':
            z_title += f'({self.normalization}-normalized)'
        plt.plot_confusion_matrix(cm=cm,
                            process_labels=self.process_labels,
                            class_labels=self.class_labels,
                            save_path=self.output().abspath,
                            normalized= self.normalization != 'None',
                            title=self.plot_title,
                            colormap=self.colormap,
                            cmap_label= z_title,
                            digits=self.digits,
                            skip_uncertainties=self.skip_uncertenties
                            )