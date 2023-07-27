import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import mplhep as hep
import re


#matplotlib.rc('lines', usetex=True) #use latex for text

#Define a CF custom color map
import matplotlib.colors as colors
cf_yellow_cmap = colors.ListedColormap(['#003675',
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
                          classes, 
                          normalize=False, 
                          title='Confusion matrix', 
                          cmap=cf_cmap,
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
    cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
    title += ' (normalized)'

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
        return "{} \n \u00B1{}".format(fmt(values[i][j]), fmt(uncs[i][j]))

  plt.style.use(hep.style.CMS)
  plt.imshow(values, interpolation='nearest', cmap=cmap)

  #Remove Major ticks and edit minor ticks
  minor_tick_length = max(int(120/len(classes)), 12)
  minor_tick_width = max(6/len(classes), 0.6)
  tick_marks = np.arange(len(classes))
  plt.tick_params(axis='both', which= 'major', 
                  bottom = False, top = False, left = False, right = False)
  plt.tick_params(axis='both', which= 'minor', 
                  bottom = True, top = True, left = True, right = True, 
                  length = minor_tick_length, width = minor_tick_width)
  plt.xticks(tick_marks + 0.5, minor=True)
  plt.yticks(tick_marks + 0.5, minor=True)
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  #Justify Color bar
  plt.colorbar(fraction=0.0471, pad=0.01, label= 'Accuracy')
  plt.clim(0,max(1, values.max()))

  #Add Matrix Elemtns
  fmt = '.3f' if normalize else '.0f'
  thresh = values.max() / 2.
  font_size = scale_font(len(classes))
  for i in range(values.shape[0]):
    for j in range(values.shape[1]):
      plt.text(j, i, value_text(i,j), fontdict={'size': font_size}, 
               horizontalalignment='center', verticalalignment='center', 
               color='white' if values[i, j] < thresh else 'black')
    #format(values[i, j], fmt)

  #Add Axes and plot labels
  hep.cms.label(llabel = 'private working', rlabel = '', )
  plt.xlabel('Predicted process', loc = 'right', labelpad= 14)
  plt.ylabel('True process', loc = 'top', labelpad= 18)

  #Saving
  plt.tight_layout()
  #plt.show()
  plt.savefig(f'./test_{type(uncs)}.png', dpi = 300, bbox_inches = 'tight')
  plt.clf()

def plot_roc_curve(input):
  pass


if __name__ == '__main__':
  from scinum import Number
  plot_confusion_matrix(np.array([[Number(i,5) for i in range(1,9)] for j in range(8)]), ['A','B','C','D','E','F','G','H'], normalize=True)
  #plot_confusion_matrix(np.array([[i for i in range(1,9)] for j in range(8)]), ['A','B','C','D','E','F','G','H'], normalize=True)