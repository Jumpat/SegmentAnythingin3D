import os

import numpy as np
import plotly.express as px
import torch
from dash import Dash, Input, Output, dcc, html
from dash.exceptions import PreventUpdate

def interactive_prompting(sam_pred, ctx, init_rgb):
    # print(ctx)
    def query(points):
        with torch.no_grad():
            input_point = points
            input_label = np.ones(len(input_point))
            masks, scores, logits = sam_pred.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

        if len(masks) == 1:
            fig1 = px.imshow(masks[0], title='mask0')
            fig2 = px.imshow(np.zeros_like(masks[0]), title='mask1')
            fig3 = px.imshow(np.zeros_like(masks[0]), title='mask2')
        elif len(masks) == 2:
            fig1 = px.imshow(masks[0], title='mask0')
            fig2 = px.imshow(masks[1], title='mask1')
            fig3 = px.imshow(np.zeros_like(masks[0]), title='mask2')
        else:
            fig1 = px.imshow(masks[0][...,None].repeat(3, -1), title='mask0')
            fig2 = px.imshow(masks[1][...,None].repeat(3, -1), title='mask1')
            fig3 = px.imshow(masks[2][...,None].repeat(3, -1), title='mask2')

        return  masks, fig1, fig2, fig3, 'You clicked: x: {}, y: {}'.format(points[:, 0], points[:, 1])
    
    _, fig1, fig2, fig3, desc = query(np.array([[100, 100], [101, 101]]))
    app = Dash('dash_app')
    app.layout = html.Div([
        html.H3(id='desc', children=desc),

        html.Div([
            html.Label('Please select the mask you want to use:'),
            dcc.RadioItems(['mask0', 'mask1', 'mask2'], id='sel_mask_id', value=None)
        ], style={'display': 'flex'}),

        html.Label('Number of prompts: '),
        html.Div([
            dcc.RadioItems(
                id='num_prompts', 
                options = [{'label': '1', 'value': 1},
                        {'label': '2', 'value': 2},
                        {'label': '3', 'value': 3}],
                value = 1),], style={'display': 'inline-block', 'width': '15%'}),

        html.Div([
            dcc.Graph(id='main_image', figure=px.imshow(init_rgb, 
                            title='original_image', aspect='equal'))
        ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),

        html.Div([
            dcc.Graph(id='selected_mask', figure=px.imshow(np.zeros_like(init_rgb), 
                            title='selected_mask', aspect='equal'))
        ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),

        html.Div([
            dcc.Graph(id='mask0', figure=fig1)
        ], style={'display': 'inline-block', 'width': '33%'}),

        html.Div([
            dcc.Graph(id='mask1', figure=fig2)
        ], style={'display': 'inline-block', 'width': '33%'}),

        html.Div([
            dcc.Graph(id='mask2', figure=fig3)
        ], style={'display': 'inline-block', 'width': '33%'}),
    ])

    @app.callback(
        Output('mask0', 'figure'),
        Output('mask1', 'figure'),
        Output('mask2', 'figure'),
        Output('desc', 'children'),
        Input('main_image', 'clickData'),
        Input('num_prompts', 'value')
    )
    def update_right_side(clickData, value_prompts):
        '''
        {'points': [{'curveNumber': 0, 'x': 62, 'y': 15, 'color': {'0': 254, '1': 254, '2': 254, '3': 1}, 'colormodel': 'rgba256', 'z': {'0': 254, '1': 254, '2': 254, '3': 1}, 'bbox': {'x0': 948.03, 'x1': 948.41, 'y0': 74.01, 'y1': 74.01}}]}
        '''
        if clickData is None:
            raise PreventUpdate
        ctx['num_clicks'] += 1
        ctx['click'].append(np.array([clickData['points'][0]['x'], clickData['points'][0]['y']]))

        if ctx['num_clicks'] < value_prompts:
            raise PreventUpdate
        
        ctx['num_clicks'] = 0
        ctx['click'] = np.stack(ctx['click'])
        ctx['saved_click'] = np.stack(ctx['click'])
        masks, fig1, fig2, fig3, desc = query(ctx['click'])
        ctx['masks'] = masks
        ctx['click'] = []
        return fig1, fig2, fig3, desc


    @app.callback(
        Output("selected_mask", "figure"),
        Input("sel_mask_id", 'value')
    )
    def update_graph(radio_items):
        # # record the selected prompt and mask
        # with open(os.path.join(self.base_save_dir, "user-specific-prompt.json"), 'w') as f:
        #     prompt_dict = {
        #         "mask_id": selected_mask
        #     }
        #     json.dump(prompt_dict, f)
        # print(f"Prompt saved in {os.path.join(self.base_save_dir, 'user-specific-prompt.json')}")

        if radio_items == 'mask0':
            ctx['select_mask_id'] = 0
            return px.imshow(ctx['masks'][0][...,None].repeat(3, -1), title='you select mask0, type Ctrl+C to start train seg', aspect='equal')
        elif radio_items == 'mask1':
            ctx['select_mask_id'] = 1
            return px.imshow(ctx['masks'][1][...,None].repeat(3, -1), title='you select mask1, type Ctrl+C to start train seg', aspect='equal')
        elif radio_items == 'mask2':
            ctx['select_mask_id'] = 2
            return px.imshow(ctx['masks'][2][...,None].repeat(3, -1), title='you select mask2, type Ctrl+C to start train seg', aspect='equal')
        else:
            raise PreventUpdate

    # @app.callback(
    #     Output('button_text', 'children'),
    #     Input('submit-val', 'n_clicks'),
    # )
    # def update_output(n_clicks, value):
    #     msg = 'points are cleared!'
    #     return msg

    app.run_server(debug=False)

    return ctx['saved_click'], ctx['select_mask_id'], ctx['masks']



