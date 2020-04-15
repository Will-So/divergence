import numpy as np
import pandas as pd
from scipy.stats import entropy

from bokeh.models import ColumnDataSource, Slider, TextInput, HoverTool
from bokeh.plotting import figure
from bokeh.palettes import Category10
from bokeh.layouts import column, row
from bokeh.io import curdoc


from kl_divergence import generate_pdf, kl_divergence

N = 10_000


def initialize_distributions(N=N, min_threshold=0, max_threshold=100):
    """Hypothetical distribution of two schools. Average of 70 but differing standard deviations"""
    dataset_1 = np.random.normal(70, 10, N).clip(min_threshold, max_threshold)
    dataset_2 = np.random.normal(70, 15, N).clip(min_threshold, max_threshold)

    return dataset_1, dataset_2


p_name = 'School 1'
q_name = 'School 2'


def format_dataset(pdfs, edges):
    left, right = edges[1:], edges[:-1]

    df = pd.DataFrame(columns=['name', 'x', 'left', 'right', 'color'])
    colors = Category10[10]

    for i, pdf in enumerate(pdfs):
        # Put all of the items of length N in the temp_df firs
        temp_df = pd.DataFrame(dict(x=pdfs[pdf], left=left, right=right))
        temp_df.loc[:, 'color'] = colors[i]
        temp_df.loc[:, 'name'] = f'School {i + 1}'
        df = df.append(temp_df, sort=True)

    grouped_p = df.groupby('left').first()
    assert len(grouped_p.name.value_counts()) == 1, "All of the last values in each group should have only one name"
    p = grouped_p.x
    q = df.groupby('left').last().x
    log_value = p * np.log(p / q)  ## Multiply by 100 to make it easier to read
    log_value.name = 'bins_divergence'

    assert len(pdfs) == 2, "The Log(p/q) Logic is not yet set up for multiple distributions"
    df = df.merge(log_value, left_on='left', right_index=True)
    # df.loc[(df.name == 'q'), 'bins_divergence'] = np.NaN ## Only define bins divergence on the p values to avoid double counting

    assert np.isclose(kl_divergence(p, q), df.bins_divergence.sum() / 2,
                      atol=1e-2), "The sum of all the bin's divergence should equal to the KL Divergence"

    df.bins_divergence *= 100  ## Multiply by 100 to make it easier to read.

    return df


def make_data(d1, d2):
    start = np.minimum(d1.min(), d2.min())
    end = np.maximum(d1.max(), d2.max())


    p, q = generate_pdf(d1, d2)
    n_bins = p.shape[0]
    pdfs = dict(p=p, q=q)  # Want to do a dict here so that we have
    assert len(pdfs) == 2, "The Log(p/q) Logic is not yet set up for multiple distributions"

    edges = np.linspace(start, end, n_bins + 1)
    df = format_dataset(pdfs, edges)
    print("Made Data")

    return df


source = ColumnDataSource({'color': [], 'left': [], 'name': [], 'right':[],
                           'x': [], 'bins_divergence': []})

hover = HoverTool()

# Disable multiple tooltips to show up at the same time.
# Based on https://stackoverflow.com/questions/36434562/displaying-only-one-tooltip-when-using-the-hovertool-tool
# At some point there will be a better solution but this works pretty great.
hover.tooltips = """
    <style>
        .bk-tooltip>div:not(:first-child) {display:none;}
    </style>

    <b>log(p/q): </b> @bins_divergence<br>

"""

plot = figure(plot_height=700, plot_width=600, title="KL Divergence by Distribution",
              tools="crosshair,pan,reset,save,wheel_zoom",)

plot.add_tools(hover)


plot.quad(source=source, top='x', bottom=0, left='left', right='right', fill_alpha=0.5, legend_field='name', color='color')

plot.xaxis.axis_label = 'x'
plot.yaxis.axis_label = 'P(x)'

p_average = Slider(title=f'{p_name} Average', value=80, start=0, end=100, step =1)
p_std = Slider(title=f'{p_name} Standard Deviation', value=10, start=0, end=30, step=1)
q_average = Slider(title=f'{q_name} Average', value=70, start=0, end=100, step =1)
q_std = Slider(title=f'{q_name} Standard Deviation', value=15, start=0, end=30, step=1)


def update():
    p_hat = p_average.value
    p_sigma = p_std.value
    q_hat = q_average.value
    q_sigma = q_std.value

    # TODO consider allowing the user to change bin size as well. And N
    np.random.seed(120)  # Make the values deterministic each time the value is inputted
    d1 = np.random.normal(p_hat, p_sigma, N).clip(0, 100)
    d2 = np.random.normal(q_hat, q_sigma, N).clip(0, 100)
    print("Generated D Values")
    data = make_data(d1, d2)

    plot.title.text = f'Total KL Divergence: {round(data.bins_divergence.sum(), 2)}'
    source.data = {'color': data.color, 'left': data.left,'name': data.name,
                   'right': data.right, 'x': data.x, 'bins_divergence': data.bins_divergence}


for w in [p_average, p_std, q_average, q_std]:
    w.on_change('value', lambda attr, old, new: update())

inputs = column(row(p_average, q_average), row(p_std, q_std))

update()
# Based on https://towardsdatascience.com/data-visualization-with-bokeh-in-python-part-ii-interactions-a4cf994e2512
curdoc().add_root(column(inputs, plot, width=800))


