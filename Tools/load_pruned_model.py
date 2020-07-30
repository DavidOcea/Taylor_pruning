# lode weight save new weight
import sys
import torch
import json

# from models.refinedet_resnet18_pruned_fixed_arch import RefineResnet18
def make_model(weight_path,in_config,all_gatelayer_info,model_save):
 
    f = open(in_config,'r')
    load_dic = json.load(f)
    block_s = load_dic['blocks']
    feature_max = load_dic['feature_mix_layer']
    idx_list = [1,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]  #有效层
    
    checkpoint = torch.load(weight_path, map_location='cpu') #map_location='cpu',默认不写gpu
    state_dict = checkpoint['state_dict']
    reminded = {}
    results = {}
    inverted_out = {}
    needs_add = {}
    depth_out = {}
    out_channels = {}

    for _, v in enumerate(state_dict.items()):
        if 'blocks.0' in v[0] and 'gate' in v[0]:
            depth_out[0] = state_dict['module.basemodel.blocks.'+str(0)+'.mobile_inverted_conv.depth_conv.gate.weight'].sum().item()
            block_s[0]['mobile_inverted_conv']['depth_out'] = int(depth_out[0])
            block_s[0]['mobile_inverted_conv']['inverted_out'] = block_s[0]['mobile_inverted_conv']['in_channels']
            block_s[0]['mobile_inverted_conv']['flag_channels'] = True
        for idx in idx_list:
            if idx == 0:
                continue
            if 'blocks.'+str(idx) in v[0] and 'bottleneck.gate' in v[0]:
                # import pdb
                # pdb.set_trace()
                reminded[idx] = state_dict['module.basemodel.blocks.'+str(idx)+'.mobile_inverted_conv.inverted_bottleneck.gate.weight'].sum().item() 

                block_s[idx]['mobile_inverted_conv']['inverted_out'] = int(reminded[idx])
                block_s[idx]['mobile_inverted_conv']['depth_out'] = int(reminded[idx])

                block_s[idx]['mobile_inverted_conv']['flag_channels'] = True
        #如果feature_mix_layer 减的话
        # if 'feature_mix_layer' in v[0]:
        #     reminded['out_channels'] = state_dict['module.basemodel.feature_mix_layer.gate.weight'].sum().item()
        #     feature_max['out_channels'] = int(reminded['out_channels'])
    
    
    lines = json.dumps(load_dic,ensure_ascii=False, indent=1)  #indent,换行存
    f.close()
    re_save_moel_weitht(weight_path,all_gatelayer_info,reminded,inverted_out,needs_add,idx_list,model_save)
    return lines

# 所减的最小层 一句gate层的分配而定              #2     #4                                                                                   
# gate_dic = {0:[0],1:[1],3:[3],5:[5],6:[7],7:[9],8:[11],9:[13],10:[15],11:[17],12:[19],13:[21],14:[23],15:[25],16:[27],17:[29],18:[31],19:[33],20:[35],21:[37]}
gate_dic = {0:[0],1:[1],3:[3],5:[5],6:[7],7:[9],8:[11],9:[13],10:[13],11:[15],12:[17],13:[19],14:[19],15:[21],16:[23],17:[25],18:[27],19:[29],20:[31],21:[33]}
# gate_dic = {0:[0],1:[1],2:[3],3:[5],4:[7],5:[9],6:[11],7:[13],8:[15],9:[17],10:[19],11:[21],12:[23],13:[25],14:[27],15:[29],16:[31],17:[33],21:[35]}
  
def re_save_moel_weitht(weight_path,all_gatelayer_info,reminded,inverted_out,needs_add_re,idx_list,model_save):
    checkpoint = torch.load(weight_path, map_location='cpu') #map_location='cpu',默认不写gpu
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    gate_count = 0
    reminded_flag = 0
    reminded_cunt = 0
    tem_ids = 0
    #如果整除通道对不上需要手动修改
    # needs_add_re[21] = 453.0
    #如果修改feature_mix:同时修改最后fc
    num_task = 5
    # import pdb
    # pdb.set_trace()
    import copy
    all_gatelayer_info_tem = copy.deepcopy(all_gatelayer_info)
    for k,v in state_dict.items():
        vs = copy.deepcopy(v)
        if 'gate' in k:
            print("gate_layer:",k,":",v.sum())
        if 'blocks' in k:
            my_str = k[22:29]
            ids = int("".join(list(filter(str.isdigit,  my_str ))))

            #第ids层不减
            if ids == 9 or ids == 13 or ids == 0:
                new_state_dict[k] = vs
                print(k,':',vs.shape)
                continue

            if ids == 0:
                gate_count = 0
                P_falg = part0(all_gatelayer_info,ids,k,vs,{0:0},{0:0},{0:0},gate_count,new_state_dict,all_gatelayer_info_tem)
            elif ids in idx_list:
                if tem_ids != ids:
                    reminded_flag2 = True
                    tem_ids = ids
                else:
                    reminded_flag2 = False

                if 'inverted_bottleneck.gate' in k:
                    reminded_flag = 1

                # if ids ==21:
                #     import pdb
                #     pdb.set_trace()
                for i in gate_dic[ids]:
                    gate_count = i
                    try:
                        p_flag = part0(all_gatelayer_info,ids,k,vs,reminded,inverted_out,needs_add_re,gate_count,new_state_dict,all_gatelayer_info_tem,reminded_flag,reminded_flag2)  
                    except:
                        import pdb
                        pdb.set_trace()
                if reminded_flag == 1:
                    reminded_cunt += 1
                if reminded_cunt >= 5:
                    reminded_cunt = 0
                    reminded_flag = 0
        #feature_mix_layer层如果减的话：需要调试
        # elif 'feature_mix_layer' in k:
        #     if 'bn.weight' in k or 'bn.bias' in k or 'bn.running_mean' in k or 'bn.running_var' in k:
        #         put = 'mid'
        #         pruned_weights = get_pruned_weights(all_gatelayer_info,0,vs,{0:0},{0:0},{0:0},put,39,0,False,all_gatelayer_info_tem)
        #         print('After pruned layer:({}) size is :{}'.format(k,pruned_weights.size()))
        #         new_state_dict[k] = pruned_weights
        #     if 'conv.weight' in k:
        #         put = 'out'
        #         pruned_weights = get_pruned_weights(all_gatelayer_info,0,vs,{0:0},{0:0},{0:0},put,39,0,False,all_gatelayer_info_tem)
        #         print('After pruned layer:({}) size is :{}'.format(k,pruned_weights.size()))
        #         new_state_dict[k] = pruned_weights
        # elif 'fcs' in k:
        #     for i_cls in range(num_task):
        #         if str(i_cls)+'.weight' in k:
        #             put = 'fcs'
        #             pruned_weights = get_pruned_weights(all_gatelayer_info,0,vs,{0:0},{0:0},{0:0},put,39,0,False,all_gatelayer_info_tem)
        #             print('After pruned layer:({}) size is :{}'.format(k,pruned_weights.size()))
        #             new_state_dict[k] = pruned_weights
        #         elif str(i_cls)+'.bias' in k:
        #             new_state_dict[k] = vs
        #             print(k,':',vs.shape)

        else:
            new_state_dict[k] = vs
            print(k,':',vs.shape)
    torch.save({
            'state_dict': new_state_dict,
        },model_save)
            

def part0(all_gatelayer_info,ids,k,v,reminded,inverted_out,needs_add,gate_count,new_state_dict,all_gatelayer_info_tem,reminded_flag=0,reminded_flag2=False):

    p_flag3 = part2(all_gatelayer_info,ids,k,v,reminded,inverted_out,needs_add,gate_count,new_state_dict,all_gatelayer_info_tem,reminded_flag,reminded_flag2)
    p_flag2 = part3(all_gatelayer_info,ids,k,v,reminded,inverted_out,needs_add,gate_count,new_state_dict,all_gatelayer_info_tem,reminded_flag,reminded_flag2)
    p_flag1 = part1(all_gatelayer_info,ids,k,v,reminded,inverted_out,needs_add,gate_count,new_state_dict)
    p_flag = p_flag1 or p_flag2 or p_flag3
    return p_flag            
   
def part1(all_gatelayer_info,ids,k,v,reminded,inverted_out,needs_add,gate_count,new_state_dict):
    if 'point_linear.bn.weight' in k or 'point_linear.bn.bias'in k or 'point_linear.bn.running_mean' in k or 'point_linear.bn.running_var' in k:
        new_state_dict[k] = v
        print(k,':',v.shape)
        return True
    else:
        return False

def part2(all_gatelayer_info,ids,k,v,reminded,inverted_out,needs_add,gate_count,new_state_dict,all_gatelayer_info_tem,reminded_flag,reminded_flag2):
    if 'inverted_bottleneck.conv.weight' in k:
        put = 'out'
        pruned_weights = get_pruned_weights(all_gatelayer_info,ids,v,reminded,inverted_out,needs_add,put,gate_count,reminded_flag,reminded_flag2,all_gatelayer_info_tem)
        print('After pruned layer:({}) size is :{}'.format(k,pruned_weights.size()))
        new_state_dict[k] = pruned_weights
        return True
    elif 'inverted_bottleneck.bn.weight' in k:
        put = 'mid'
        pruned_weights = get_pruned_weights(all_gatelayer_info,ids,v,reminded,inverted_out,needs_add,put,gate_count,reminded_flag,reminded_flag2,all_gatelayer_info_tem)
        print('After pruned layer:({}) size is :{}'.format(k,pruned_weights.size()))
        new_state_dict[k] = pruned_weights
        return True
    elif 'inverted_bottleneck.bn.bias' in k:
        put = 'mid'
        pruned_weights = get_pruned_weights(all_gatelayer_info,ids,v,reminded,inverted_out,needs_add,put,gate_count,reminded_flag,reminded_flag2,all_gatelayer_info_tem)
        print('After pruned layer:({}) size is :{}'.format(k,pruned_weights.size()))
        new_state_dict[k] = pruned_weights
        return True
    elif 'inverted_bottleneck.bn.running_mean' in k:
        put = 'mid'
        pruned_weights = get_pruned_weights(all_gatelayer_info,ids,v,reminded,inverted_out,needs_add,put,gate_count,reminded_flag,reminded_flag2,all_gatelayer_info_tem)
        print('After pruned layer:({}) size is :{}'.format(k,pruned_weights.size()))
        new_state_dict[k] = pruned_weights
        return True
    elif 'inverted_bottleneck.bn.running_var' in k:
        put = 'mid'
        pruned_weights = get_pruned_weights(all_gatelayer_info,ids,v,reminded,inverted_out,needs_add,put,gate_count,reminded_flag,reminded_flag2,all_gatelayer_info_tem)
        print('After pruned layer:({}) size is :{}'.format(k,pruned_weights.size()))
        new_state_dict[k] = pruned_weights
        return True
    else:
        return False
    

def part3(all_gatelayer_info,ids,k,v,reminded,inverted_out,needs_add,gate_count,new_state_dict,all_gatelayer_info_tem,reminded_flag,reminded_flag2):
    if 'depth_conv.conv.weight' in k:
        put = 'out'
        pruned_weights = get_pruned_weights(all_gatelayer_info,ids,v,reminded,inverted_out,needs_add,put,gate_count,reminded_flag,reminded_flag2,all_gatelayer_info_tem)
        print('After pruned layer:({}) size is :{}'.format(k,pruned_weights.size()))
        new_state_dict[k] = pruned_weights
        return True
    elif 'depth_conv.bn.weight' in k:
        put = 'mid'
        pruned_weights = get_pruned_weights(all_gatelayer_info,ids,v,reminded,inverted_out,needs_add,put,gate_count,reminded_flag,reminded_flag2,all_gatelayer_info_tem)
        print('After pruned layer:({}) size is :{}'.format(k,pruned_weights.size()))
        new_state_dict[k] = pruned_weights
        return True
    elif 'depth_conv.bn.bias' in k:
        put = 'mid'
        pruned_weights = get_pruned_weights(all_gatelayer_info,ids,v,reminded,inverted_out,needs_add,put,gate_count,reminded_flag,reminded_flag2,all_gatelayer_info_tem)
        print('After pruned layer:({}) size is :{}'.format(k,pruned_weights.size()))
        new_state_dict[k] = pruned_weights
        return True
    elif 'depth_conv.bn.running_mean' in k:
        put = 'mid'
        pruned_weights = get_pruned_weights(all_gatelayer_info,ids,v,reminded,inverted_out,needs_add,put,gate_count,reminded_flag,reminded_flag2,all_gatelayer_info_tem)
        print('After pruned layer:({}) size is :{}'.format(k,pruned_weights.size()))
        new_state_dict[k] = pruned_weights
        return True
    elif 'depth_conv.bn.running_var' in k:
        put = 'mid'
        pruned_weights = get_pruned_weights(all_gatelayer_info,ids,v,reminded,inverted_out,needs_add,put,gate_count,reminded_flag,reminded_flag2,all_gatelayer_info_tem)
        print('After pruned layer:({}) size is :{}'.format(k,pruned_weights.size()))
        new_state_dict[k] = pruned_weights
        return True
    elif 'point_linear.conv.weight' in k:
        put = 'in'
        pruned_weights = get_pruned_weights(all_gatelayer_info,ids,v,reminded,inverted_out,needs_add,put,gate_count,reminded_flag,reminded_flag2,all_gatelayer_info_tem)
        print('After pruned layer:({}) size is :{}'.format(k,pruned_weights.size()))
        new_state_dict[k] = pruned_weights
        return True
    else:
        return False
    
def get_pruned_weights(all_gatelayer_info,index,v,reminded,inverted_out,needs_add,put,gate_count,reminded_flag,reminded_flag2,all_gatelayer_info_tem):
    pruned_layer_zero = []
    for j in range(0,len(all_gatelayer_info_tem[gate_count])):
        if int(all_gatelayer_info_tem[gate_count][j]) == 0:
            pruned_layer_zero.append(j)
    # import pdb
    # pdb.set_trace()
    if put =='in':
        v[:,pruned_layer_zero,:,:]*=0.0
    if put == 'out':
        v[pruned_layer_zero,:,:,:]*=0.0
    if put == 'mid':
        v[pruned_layer_zero]*=0.0
    if put == 'fcs':
        v[:,pruned_layer_zero]*=0.0

    pruned_layer_weights_temp = []
    for i in range(0,len(all_gatelayer_info[gate_count])):
        if int(all_gatelayer_info[gate_count][i]) == 1:
            pruned_layer_weights_temp.append(i)
            
    if put =='in':
        pruned_weights = v[:,pruned_layer_weights_temp,:,:]
    if put == 'out':
        pruned_weights = v[pruned_layer_weights_temp,:,:,:]
    if put == 'mid':
        pruned_weights = v[pruned_layer_weights_temp]
    if put == 'fcs':
        pruned_weights = v[:,pruned_layer_weights_temp]
    return pruned_weights


def read_pruned_weights(weight_path):
    checkpoint = torch.load(weight_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    # #输出gate结构
    for k, v in state_dict.items():
        if 'gate' in k:
            print(k,':',v.sum())
        print(k,":",v.shape)
    # return
    
    all_gatelayer_info = []
    for k, v in state_dict.items():
        count = 0
        if 'gate' in k:
            for i in range(len(v)):
                if int(v[i]) == 1:
                    count += 1
            all_gatelayer_info.append(v)
    return all_gatelayer_info

def main():
    weight_path = '../runs/mult_5T/mult_prun8_gpu/models/21.pth.tar'
    
    #sava model_weight
    all_gatelayer_info = read_pruned_weights(weight_path)
    # re_save_moel_weitht(weight_path,all_gatelayer_info,reminded=0,inverted_out=120,needs_add=2)
    
    #save model_config
    in_config = './net_config.json'
    out_config = './configs/purned_config_717_t2_tem.json'
    model_save = './models/purn_20200717_5T_t_20e.pth.tar'
    with open(out_config, 'w') as f:
        f.writelines(make_model(weight_path,in_config,all_gatelayer_info,model_save))
    

    
if __name__ == "__main__":
    main()
