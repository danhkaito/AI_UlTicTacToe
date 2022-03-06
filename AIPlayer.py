import numpy as np
from state import *

count_in_center=0
index_diff_block=-1
opposite_block=-1


def eval(cur_state:State):
    global_point=np.zeros(9)
    i=0
    for block in cur_state.blocks:
        point=0
        countX_row=-np.count_nonzero(block==1, axis=1)
        countX_col=-np.count_nonzero(block==1, axis=0)
        countO_row=np.count_nonzero(block==-1, axis=1)
        countO_col=np.count_nonzero(block==-1, axis=0)
        for i in range(0,3):
            if (countX_row[i]<0) and (countO_row[i]>0):
                continue
            else:
                point+=(countX_row[i]+countO_row[i])
        for i in range(0,3):
            if (countX_col[i]<0) and (countO_col[i]>0):
                continue
            else:
                point+=(countX_col[i]+countO_col[i])
        countX_diagnol_topright=0
        countO_diagnol_topright=0
        countX_diagnol_topleft=0
        countO_diagnol_topleft=0
        for i in range (0,3):
            if(block[i][i]==1):
                countX_diagnol_topleft-=1
            elif (block[i][i]==-1):
                countO_diagnol_topleft+=1
            if(block[i][2-i]==1):
                countX_diagnol_topright-=1
            elif (block[i][2-i]==-1):
                countO_diagnol_topright+=1
        if (countX_diagnol_topleft)==0 or (countO_diagnol_topleft)==0:
            point+=(countX_diagnol_topleft+countO_diagnol_topleft)
        if (countX_diagnol_topright)==0 or (countO_diagnol_topright)==0:
            point+=(countX_diagnol_topright+countO_diagnol_topright)
        global_point[i]=point
        i+=1

    return global_point.sum()





def max_value(cur_state, alpha, beta, depth):
    leaf_state_val=terminate_state(cur_state, depth)
    if leaf_state_val!=None:
        return leaf_state_val
    else:
        v=-np.inf
        valid_moves=cur_state.get_valid_moves
        for move in valid_moves:
            temp_state=State(cur_state)
            temp_state.free_move=cur_state.free_move
            temp_state.act_move(move)
            val=min_value(temp_state, alpha, beta,depth-1)
            v=max(v,val)
            if(v>=beta):
                return v
            alpha=max(alpha, v)
    return v


def min_value(cur_state, alpha, beta, depth):
    leaf_state_val=terminate_state(cur_state, depth)
    if leaf_state_val!=None:
        return leaf_state_val
    else:
        v=np.inf
        valid_moves=cur_state.get_valid_moves
        if(len(valid_moves)!=0):
            for move in valid_moves:
                temp_state=State(cur_state)
                temp_state.free_move=cur_state.free_move
                temp_state.act_move(move)
                val=max_value(temp_state, alpha, beta,depth-1)
                v=min(v,val)
                if(v<=alpha):
                    return v
                beta=min(beta, v)
    return v


def terminate_state(cur_state, depth):
    if(depth==0):
        return eval(cur_state)
    else: 
        result=cur_state.game_result(cur_state.global_cells.reshape(3,3))
        if(result!=None):
            if(result==0): return 0
            return -np.inf*result
        else:
            return None




def minimax_ab_cutoff(cur_state, tree_depth):
    alpha=-np.inf
    beta=np.inf
    v=-np.inf
    valid_moves=cur_state.get_valid_moves
    if(len(valid_moves)!=0):
        optimal_move=valid_moves[0]
        for move in valid_moves:
            temp_state=State(cur_state)
            temp_state.free_move=cur_state.free_move
            temp_state.act_move(move)
            new_val=min_value(temp_state, alpha, beta, tree_depth)
            if new_val>v:
                v=new_val
                alpha=v
                optimal_move=move
        return optimal_move



def select_move(cur_state, remain_time):
    global index_diff_block
    global count_in_center
    global opposite_block
    valid_moves = cur_state.get_valid_moves
    ##Go first
    if(cur_state.player_to_move==1):
        if(cur_state.previous_move==None):
            count_in_center=0
            index_diff_block=-1
            opposite_block=-1
            return UltimateTTT_Move(4,1,1,cur_state.player_to_move)
        elif (index_diff_block==-1):
            index_valid_block=cur_state.previous_move.x*3+cur_state.previous_move.y
            if(count_in_center<7):
                count_in_center+=1
                
                return UltimateTTT_Move(index_valid_block, 1,1, cur_state.player_to_move)
            else:
                index_diff_block=index_valid_block
                opposite_block=8-index_diff_block
                return UltimateTTT_Move(index_diff_block, cur_state.previous_move.x,cur_state.previous_move.y, cur_state.player_to_move)
        else:
            if(cur_state.free_move==False):
                if(cur_state.blocks[valid_moves[0].index_local_board][int(index_diff_block/3)][index_diff_block%3]==0):
                    return UltimateTTT_Move(valid_moves[0].index_local_board,int(index_diff_block/3),index_diff_block%3, cur_state.player_to_move)
                else:
                    return UltimateTTT_Move(valid_moves[0].index_local_board,int((8-index_diff_block)/3),(8-index_diff_block)%3, cur_state.player_to_move)
            if(cur_state.free_move==True):
                if(cur_state.blocks[opposite_block][int(index_diff_block/3)][index_diff_block%3]==0):
                    return UltimateTTT_Move(opposite_block,int(index_diff_block/3),index_diff_block%3, cur_state.player_to_move)
                else:
                    return UltimateTTT_Move(opposite_block,int((8-index_diff_block)/3),(8-index_diff_block)%3, cur_state.player_to_move)
    #Go second
    else:
        return minimax_ab_cutoff(cur_state, 4)
    return None
