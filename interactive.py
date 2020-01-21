import numpy as np
import pandas as pd
from scipy.stats import entropy

from bokeh.models import ColumnDataSource, Slider, TextInput, HoverTool
from bokeh.plotting import figure
from bokeh.palettes import Category20

from .kl_divergence import generate_pdf, kl_divergence

N = 10_000

def initialize_distributions(N=N):
    dataset_1 = np.random.normal(100, 20, N)
    dataset_2 = np.random.normal(100, 25, N)

    return dataset_1, dataset_2

d1, d2 = initialize_distributions()
p, q = generate_pdf(d1, d2)
pdfs = dict(p=p, q=q)  # Want to do a dict here so that we have
assert len(pdfs) == 2, "The Log(p/q) Logic is not yet set up for multiple distributions"

pdfs = dict(p=p, q=q)  # Want to do a dict here so that we have


def format_dataset(pdfs, edges):
    left, right = edges[1:], edges[:-1]

    df = pd.DataFrame(columns=['name', 'x', 'left', 'right', 'color'])
    colors = Category20[10]

    for i, pdf in enumerate(pdfs):
        # Put all of the items of length N in the temp_df firs
        temp_df = pd.DataFrame(dict(x=pdfs[pdf], left=left, right=right))
        temp_df.loc[:, 'color'] = colors[i]
        temp_df.loc[:, 'name'] = pdf
        df = df.append(temp_df, sort=True)

    grouped_p = df.groupby('left').first()
    assert len(grouped_p.name.value_counts()) == 1, "All of the last values in each group should have only one name"
    assert grouped_p.name.iloc[0] == 'p', "First group should be p"
    p = grouped_p.x
    q = df.groupby('left').last().x
    log_value = p * np.log(p / q)  ## Multiply by 100 to make it easier to read
    log_value.name = 'bins_divergence'

    df = df.merge(log_value, left_on='left', right_index=True)

    assert np.isclose(entropy(p, q), df.bins_divergence.sum() / 2,  # Divide by 2 because we double count since both p and q calculate p/q
                      atol=1e-2), "The sum of all the bin's divergence should equal to 1"

    df.bins_divergence *= 100  ## Multiply by 100 to make it easier to read.

    return df

n_bins = p.shape[0]
start = np.minimum(d1.min(), d2.min())
end = np.maximum(d1.max(), d2.max())

edges = np.linspace(start, end, n_bins + 1)

df = format_dataset(pdfs, edges)
data = ColumnDataSource(df)


plot = figure(plot_height=700, plot_width=700, title="KL Divergence by Distribution",
              tools="crosshair,pan,reset,save,wheel_zoom",)
plot.quad(source=data, top='x', bottom=0, left='left', right='right', fill_alpha=0.5, legend_field='name', color='color')

hover = HoverTool()

# Based on https://stackoverflow.com/questions/36434562/displaying-only-one-tooltip-when-using-the-hovertool-tool
# At some point there will be a better solution but this works pretty great.
hover.tooltips = """
    <style>
        .bk-tooltip>div:not(:first-child) {display:none;}
    </style>

    <b>log(p/q): </b> @bins_divergence<br>

"""

plot.xaxis.axis_label = 'x'
plot.yaxis.axis_label = 'P(x)'

# Based on https://towardsdatascience.com/data-visualization-with-bokeh-in-python-part-ii-interactions-a4cf994e2512


