import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

def run_sensitivity_analysis(model, t_dim, feature_names, device, save_path):
    model.eval()
    print("Running Intervention Analysis...")
    
    # 기준점: Group 0 (Control)
    t_ctrl = torch.tensor([0]).to(device)
    t_ctrl_oh = F.one_hot(t_ctrl, num_classes=t_dim).float()
    
    with torch.no_grad():
        m_base = model.mechanism_net(t_ctrl_oh).cpu().numpy().flatten()
        results = []
        
        # T=1 ~ T=N까지 개입
        for i in range(1, t_dim):
            t_treat = torch.tensor([i]).to(device)
            t_treat_oh = F.one_hot(t_treat, num_classes=t_dim).float()
            
            m_pred = model.mechanism_net(t_treat_oh).cpu().numpy().flatten()
            
            # 차이 계산
            delta = m_pred - m_base
            
            for f_idx, val in enumerate(delta):
                results.append({
                    'Condition_ID': i,
                    'Feature': feature_names[f_idx],
                    'Importance': abs(val), # 절대값(중요도)
                    'Raw_Change': val       # 실제 변화량(+/-)
                })
                
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    
    # 랭킹 출력
    avg_rank = df.groupby('Feature')['Importance'].mean().sort_values(ascending=False)
    print("\n[Top Features by Causal Effect]")
    print(avg_rank.head(5))
    