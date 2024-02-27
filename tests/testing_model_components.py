model_name = f'dwnsamp_{var}'
model_outdir = f'{log_dir}/{model_name}/'
Path(model_outdir).mkdir(parents=True, exist_ok=True)

if var=='PRECIP': datgen.load_EA_rain_gauge_data()

#specify_chkpnt = None # f'{model_name}/checkpoint.pth' 
specify_chkpnt = f'{model_name}/checkpoint.pth'

## dummy batch for model param fetching
batch = datgen.get_batch(var, batch_size=train_pars.batch_size,
                         batch_type='train')
                              
## create model              
model = SimpleDownscaler(
    input_channels=batch['coarse_inputs'].shape[1],
    hires_fields=batch['fine_inputs'].shape[1],
    output_channels=batch['coarse_inputs'].shape[1],
    context_channels=batch['station_data'][0].shape[1],
    filters=model_pars.filters,
    dropout_rate=model_pars.dropout_rate,
    scale=data_pars.scale,
    scale_factor=model_pars.scale_factor,
    attn_heads=model_pars.attn_heads,
    ds_cross_attn=model_pars.ds_cross_attn,
    pe=model_pars.pe
 )

del(batch)
model.to(device)

## load checkpoint
model, opt, chk = setup_checkpoint(model, None, device, load_prev_chkpnt,
                                   model_outdir, log_dir,
                                   specify_chkpnt=specify_chkpnt,
                                   reset_chkpnt=reset_chkpnt)
model.eval()

# get tile(s) of whole UK
date_string = "20140101"
it = 9
tile = False
context_frac = 0.7
constraints = False
batch = datgen.get_all_space(var, batch_type='train',
                             context_frac=context_frac,
                             date_string=date_string, it=it,
                             timestep='hourly',
                             tile=tile,
                             return_constraints=constraints)
ixs = batch['ixs']
iys = batch['iys']

batch = Batch(batch, var_list=var, device=device, constraints=constraints)

masks_250 = create_attention_masks(model, batch, var,
                               dist_lim = 250, #80,
                               dist_lim_far = None,
                               attn_eps = None,
                               poly_exp = 3.,
                               diminish_model = None,
                               dist_pixpass = 125,
                               pass_exp = 1.5)
masks_80 = create_attention_masks(model, batch, var,
                               dist_lim = 60, #80,
                               dist_lim_far = None,
                               attn_eps = None,
                               poly_exp = 3.,
                               diminish_model = None,
                               dist_pixpass = 125,
                               pass_exp = 1.5)
                               
                               
                                   
import torch
import torch.nn as nn
from layers import find_num_pools, GridPointAttn

coarse_inputs = batch.coarse_inputs.clone()
fine_inputs = batch.fine_inputs.clone()
context_data = batch.context_data.copy()
context_locs = batch.context_locs.copy()         

context_masks_250 = masks_250['context_masks'].copy()
context_soft_masks_250 = masks_250['context_soft_masks'].copy()
pixel_passers_250 = masks_250['pixel_passers'].copy()

context_masks_80 = masks_80['context_masks'].copy()
context_soft_masks_80 = masks_80['context_soft_masks'].copy()
pixel_passers_80 = masks_80['pixel_passers'].copy()


coarse_size = coarse_inputs.shape[-1]
hires_size = fine_inputs.shape[-1]
sizes = [coarse_size * (model.scale_factor**i) for i in range(1, model.nups+1)]
final_scale = coarse_size * model.scale / sizes[-1]

sf = nn.functional.interpolate(fine_inputs, scale_factor=1./model.scale, mode='bilinear')
x = torch.cat([coarse_inputs, sf], dim=1)


x_80 = model.grid_point_attn_0(x, context_data, context_locs, hires_size,
                               mask=context_masks_80[0],
                               softmask=context_soft_masks_80[0],
                               pixel_passer=pixel_passers_80[0])

x_250 = model.grid_point_attn_0(x, context_data, context_locs, hires_size,
                                mask=context_masks_250[0],
                                softmask=context_soft_masks_250[0],
                                pixel_passer=pixel_passers_250[0])


# we have x_250 == x_80!!
# dig into grid_point_attn_0 layer
raw_size = hires_size

mask_80 = context_masks_80[0]
softmask_80 = context_soft_masks_80[0]
pixel_passer_80 = pixel_passers_80[0]

mask_250 = context_masks_250[0]
softmask_250 = context_soft_masks_250[0]
pixel_passer_250 = pixel_passers_250[0]

    if mask_80 is None: mask_80 = [None] * len(context_data)
    if softmask_80 is None: softmask_80 = [None] * len(context_data)
    if pixel_passer_80 is None: pixel_passer_80 = [None] * len(context_data)
    
    if mask_250 is None: mask_250 = [None] * len(context_data)
    if softmask_250 is None: softmask_250 = [None] * len(context_data)
    if pixel_passer_250 is None: pixel_passer_250 = [None] * len(context_data)
        
    x = model.grid_point_attn_0.embed_grid(x)
    attn_out = []
    raw_H = x.shape[2]
    raw_W = x.shape[3]

    for b in range(len(context_data)):
        if context_data[b].shape[-1]==0: # 0 as we have removed null tags
            attn_out.append(x[b])
        else:      
            # take batch slice and reshape to 1D vector list
            x_b = x[b:(b+1)]
            x_b = torch.reshape(x_b, (x_b.shape[0], x_b.shape[1], raw_H*raw_W))
            
            # embed context data and do off-grid positional encoding 
            context_b = model.grid_point_attn_0.embed_context(context_data[b].transpose(-2,-1)).transpose(-2,-1)
    
            # calculate cross-attention
            x_b = x_b.transpose(-2,-1)

            x_b_80 = model.grid_point_attn_0.sublayer(x_b, 
                lambda x_b: model.grid_point_attn_0.cross_attn(x_b,
                                            context_b.transpose(-2,-1),
                                            context_b.transpose(-2,-1),
                                            mask_80[b],
                                            softmask_80[b],
                                            pixel_passer_80[b])
            )
            x_b_250 = model.grid_point_attn_0.sublayer(x_b, 
                lambda x_b: model.grid_point_attn_0.cross_attn(x_b,
                                            context_b.transpose(-2,-1),
                                            context_b.transpose(-2,-1),
                                            mask_250[b],
                                            softmask_250[b],
                                            pixel_passer_250[b])
            )
            
            ## we have x_b_80 == x_b_250
            ## dig into model.grid_point_attn_0.cross_attn
            
            
            # def forward(self, query, key, value,
            # mask=None, softmask=None, pixel_passer=None):
            query = x_b
            key = context_b.transpose(-2,-1)
            value = context_b.transpose(-2,-1)
            
            mask250 = None
            softmask250 = softmask_250[b].clone()
            pixel_passer250 = pixel_passer_250[b].clone()
            
            mask80 = None
            softmask80 = softmask_80[b].clone()
            pixel_passer80 = pixel_passer_80[b].clone()
            
            # Same mask applied to all h heads...
            # unless we use the ALibi reducing m_factor
            if softmask250 is not None:
                # if not ALibi just unsqueeze the head dimension:
                #softmask = softmask.unsqueeze(1)
                
                # if ALibi, multiply each head by decreasng m_factor:
                sizes = list(softmask250.shape)
                sizes.insert(1, model.grid_point_attn_0.cross_attn.h)
                use_softmask250 = torch.zeros(tuple(sizes), dtype=torch.float32).to(softmask250.device)
                m_factors = [1/2**n for n in range(model.grid_point_attn_0.cross_attn.h)]
                for hh in range(model.grid_point_attn_0.cross_attn.h):
                    use_softmask250[:,hh,...] = softmask250[:,hh,...] * m_factors[hh]
                softmask250 = use_softmask250
                
            if softmask80 is not None:
                # if not ALibi just unsqueeze the head dimension:
                #softmask = softmask.unsqueeze(1)
                
                # if ALibi, multiply each head by decreasng m_factor:
                sizes = list(softmask80.shape)
                sizes.insert(1, model.grid_point_attn_0.cross_attn.h)
                use_softmask80 = torch.zeros(tuple(sizes), dtype=torch.float32).to(softmask80.device)
                m_factors = [1/2**n for n in range(model.grid_point_attn_0.cross_attn.h)]
                for hh in range(model.grid_point_attn_0.cross_attn.h):
                    use_softmask80[:,hh,...] = softmask80[:,hh,...] * m_factors[hh]
                softmask80 = use_softmask80
                
            nbatches = query.size(0)
            
            # do all the linear projections in batch from d_model => h x d_k 
            query, key, value = \
                [l(x).view(nbatches, -1, model.grid_point_attn_0.cross_attn.h, model.grid_point_attn_0.cross_attn.d_k).transpose(1, 2)
                 for l, x in zip(model.grid_point_attn_0.cross_attn.linears, (query, key, value))]
            
            # apply attention on all the projected vectors in batch 
            x, model.grid_point_attn_0.cross_attn.attn = attention(query, key, value,
                                     mask=mask, 
                                     dropout=model.grid_point_attn_0.cross_attn.dropout,
                                     softmask=softmask)
            
            # "concat" using a view and apply a final linear
            x = x.transpose(1, 2).contiguous() \
                 .view(nbatches, -1, model.grid_point_attn_0.cross_attn.h * model.grid_point_attn_0.cross_attn.d_k)
                 
            if pixel_passer is not None:
                return model.grid_point_attn_0.cross_attn.linears[-1](x) * torch.reshape(pixel_passer, (pixel_passer.shape[0], pixel_passer.shape[1]*pixel_passer.shape[2])).unsqueeze(-1)
            else:
                return model.grid_point_attn_0.cross_attn.linears[-1](x)
