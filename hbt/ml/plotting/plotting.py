import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import re


#matplotlib.rc('lines', usetex=True) #use latex for text

#Define a CF custom color map
import matplotlib.colors as colors
cf_green_cmap = colors.ListedColormap(['#212121',
 '#242723',
 '#262D25',
 '#283426',
 '#2A3A26',
 '#2C4227',
 '#2E4927',
 '#305126',
 '#325A25',
 '#356224',
 '#386B22',
 '#3B7520',
 '#3F7F1E',
 '#43891B',
 '#479418',
 '#4C9F14',
 '#52AA10',
 '#58B60C',
 '#5FC207',
 '#67cf02']) #type: ignore
cf_ygb_cmap = colors.ListedColormap(['#003675',
  '#005B83',
  '#008490',
  '#009A83',
  '#00A368',
  '#00AC49',
  '#00B428',
  '#00BC06',
  '#0CC300',
  '#39C900',
  '#67cf02',
  '#72DB02',
  '#7EE605',
  '#8DF207',
  '#9CFD09',
  '#AEFF0B',
  '#C1FF0E',
  '#D5FF10',
  '#EBFF12',
  '#FFFF14']
) # type: ignore
cf_cmap = colors.ListedColormap([
'#002C9C',
'#00419F',
'#0056A2',
'#006BA4',
'#0081A7',
'#0098AA',
'#00ADAB',
'#00B099',
'#00B287',
'#00B574',
'#00B860',
'#00BB4C',
'#00BD38',
'#00C023',
'#00C20D',
'#06C500',
'#1EC800',
'#36CA00',
'#4ECD01',
'#67cf02'
]) # type: ignore

def plot_confusion_matrix(cm, 
                          process_labels,
                          class_labels, 
                          save_path:str='./cm_plot.png',
                          normalize=False, 
                          title='Confusion matrix', 
                          cmap=cf_cmap,
                          cmap_label:str= 'Accuracy',
                          digits:int = 3):
  """
  Plots a confusion matrix.

  Args:
    cm: Confusion matrix.
    classes: List of class labels.
    normalize: Whether to normalize the confusion matrix.
    title: Title of the confusion matrix.
    cmap: Colormap.
  """
  #TODO check for unvalid inputs

  def scale_font(class_number: int) -> int:
    if class_number > 10:
      return max(8, int(-8/10 * class_number + 23))
    else:
      return int(class_number/14*(9 * class_number - 177) + 510/7)
  
    
  def get_errors(matrix):
    from scinum import UP
    get_errors = np.vectorize(lambda x: x.get(UP, unc=True))
    return get_errors(matrix)
    
  if normalize:
    cmap_label += ' (normalized)'

  #Get values and their uncertenties
  values = cm.astype(np.float32)
  if cm.dtype.name == 'object':
    uncs = get_errors(cm)
    skip_uncertainties = False
  else:
    skip_uncertainties = True
    uncs = None

  def value_text(i, j):
    def fmt(v):
        s = "{{:.{}f}}".format(digits).format(v)
        return s if re.sub(r"(0|\.)", "", s) else ("<" + s[:-1] + "1")
    if skip_uncertainties:
        return fmt(values[i][j])
    else:
        return "{}\n\u00B1{}".format(fmt(values[i][j]), fmt(uncs[i][j]))

  plt.style.use(hep.style.CMS)
  plt.imshow(values, interpolation='nearest', cmap=cmap)

  #Remove Major ticks and edit minor ticks
  minor_tick_length = max(int(120/len(class_labels)), 12)
  minor_tick_width = max(6/len(class_labels), 0.6)
  xtick_marks = np.arange(len(class_labels))
  ytick_marks = np.arange(len(process_labels))
  plt.tick_params(axis='both', which= 'major', 
                  bottom = False, top = False, left = False, right = False)
  plt.tick_params(axis='both', which= 'minor', 
                  bottom = True, top = True, left = True, right = True, 
                  length = minor_tick_length, width = minor_tick_width)
  plt.xticks(xtick_marks + 0.5, minor=True)
  plt.yticks(ytick_marks + 0.49, minor=True)
  plt.xticks(xtick_marks, class_labels, rotation=45)
  plt.yticks(ytick_marks, process_labels)

  #Justify Color bar
  plt.colorbar(fraction=0.0471, pad=0.01, label= cmap_label)
  plt.clim(0,max(1, values.max()))

  #Add Matrix Elemtns
  thresh = values.max() / 2.
  font_size = scale_font(len(class_labels))
  for i in range(values.shape[0]):
    for j in range(values.shape[1]):
      plt.text(j, i, value_text(i,j), fontdict={'size': font_size}, 
               horizontalalignment='center', verticalalignment='center', 
               color='white' if values[i, j] < thresh else 'black')
      #TODO for centering print the errors as two seperate texts
    #format(values[i, j], fmt)

  #Add Axes and plot labels
  hep.cms.label(llabel = 'private work', rlabel = title if title != None else '')
  plt.xlabel('Predicted process', loc = 'right', labelpad= 14)
  plt.ylabel('True process', loc = 'top', labelpad= 18)

  #Saving
  plt.tight_layout()
  #plt.show()
  plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
  plt.clf()


def plot_roc_curve(save_path:str='./roc_plot.png',
                   fpr=None, 
                   tpr=None,
                    *args, 
                    label:str='ROC Curve', 
                    grid:tuple=None,  # type: ignore
                    logscale:bool=False, 
                    auc_scores:dict or float=None,
                    input_dict:dict=None):
  """Creats the plot for givin ROC curve data

  Args:
      save_path (str, optional): Path to save the plot. Defaults to './roc_plot.png'.
      fpr (_type_, optional): array of list with the FPR values. Defaults to None.
      tpr (_type_, optional): array of list with the TPR values. Defaults to None.
      label (str, optional): plot title of the ROC Curve. Defaults to 'ROC Curve'
      grid (tuple, optional): The layout grid for the plots. 
                              If not specified, the number of rows and columns will be set to an optimum. 
                              Defaults to None.
      logscale (bool, optional): Sets the axis scale type to a logarithmic scale. 
                              Defaults to False.
      auc_scores (dictorfloat, optional): AUC scores for the givin ROC curve. 
                                          The parameter should be givin either as a float for a single ROC curve plot or 
                                          if `input_dict` is givin as a dictionary with the same keys as `input_dict`. 
                                          Defaults to None.
      input_dict (dict, optional): If specified, this parameter overrides `fpr`, `tpr` and `label`. 
                                    The input dictionary should have the form {plot_label:{'fpr': <<fpr array>>, 'tpr:<<tpr array>>}, ...}. 
                                    Defaults to None.

  Returns:
      _type_: None
  """  

  #TODO Check for unvalid inputs
  def get_grid(n):
    nrow = round(np.sqrt(n))
    ncol = int(n/nrow)
    while(nrow*ncol < n):
        ncol+=1
    return nrow, ncol

  def plot_roc(ax:plt.Axes, x, y, log_scale:bool, axtitle):
    ax.plot(x, y)
    ax.set_yscale('log' if log_scale else 'linear')
    ax.set_title(axtitle)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    #TODO Set X and Y Tick to be the same number
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.tick_params(axis='x', pad=10)
    ax.tick_params(axis='y', pad=10)
  
  if input_dict == None:
    input_dict={label: {'fpr': fpr, 'tpr':tpr}}
    if not isinstance(auc_scores, (dict, type(None))):
      auc_scores={label: auc_scores}

  nrow, ncol = get_grid(len(input_dict.keys())) if grid == None else grid

  plt.style.use(hep.style.CMS)
  fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(6*ncol,6*nrow+1), dpi=300)
  axs = np.array(axs).flatten()
  hep.cms.label(llabel = 'private work', rlabel = '', ax=axs[0], pad=0.1)
  for (key, item), ax in zip(input_dict.items(), axs):

    plot_roc(ax, item['fpr'], item['tpr'], logscale, f'{key} (AUC = {auc_scores[key]})' if auc_scores else key)
  #Saving
  plt.tight_layout()
  fig.savefig(save_path, dpi = 300, bbox_inches='tight')
  plt.clf()


if __name__ == '__main__':
  from scinum import Number
  for n, i in enumerate([cf_cmap]):
    plot_confusion_matrix(np.array([[Number(np.random.random(),5) for i in range(1,9)] for j in range(8)]), ['A','B','C','D','E','F','G','H'],['A','B','C','D','E','F','G','H'], title='test', normalize=True, cmap=i, save_path=f'./cmap_{n}.png')
  #plot_confusion_matrix(np.array([[i for i in range(1,9)] for j in range(8)]), ['A','B','C','D','E','F','G','H'], normalize=True)