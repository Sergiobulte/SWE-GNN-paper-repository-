# Libraries
import torch
import wandb
from wandb import Config
import PIL, cv2
import torch.optim as optim
from torch_geometric.loader import DataLoader

from utils.dataset import create_model_dataset, to_temporal_dataset
from utils.dataset import get_temporal_test_dataset_parameters
from utils.load import read_config
from utils.visualization import PlotRollout
from utils.miscellaneous import get_numerical_times, calculate_speed_ups, get_model, SpatialAnalysis
from training.train import Trainer

def fix_dict_in_config(wandb):
    config = dict(wandb.config)
    for k, v in config.copy().items():
        if '.' in k:
            new_key = k.split('.')[0]
            inner_key = k.split('.')[1]
            if new_key not in config.keys():
                config[new_key] = {}
            config[new_key].update({inner_key: v})
            del config[k]
    
    wandb.config = Config()
    for k, v in config.items():
        wandb.config[k] = v

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_parameters = config.dataset_parameters
    train_dataset_name = 'DR49_trainshow'
    test_dataset_name = 'DR49_testshow'
    scalers = config.scalers
    selected_node_features = config.selected_node_features
    selected_edge_features = config.selected_edge_features

    train_dataset, val_dataset, test_dataset, scalers = create_model_dataset(
        train_dataset_name, test_dataset_name, scalers=scalers, device=device, **dataset_parameters,
        **selected_node_features, **selected_edge_features
    )

    temporal_dataset_parameters = config.temporal_dataset_parameters

    temporal_train_dataset = to_temporal_dataset(
    train_dataset, **temporal_dataset_parameters)


    node_features, edge_features = temporal_train_dataset[0].x.size(-1), temporal_train_dataset[0].edge_attr.size(-1)
    num_nodes, num_edges = temporal_train_dataset[0].x.size(0), temporal_train_dataset[0].edge_attr.size(0)

    previous_t = temporal_dataset_parameters['previous_t']
    test_size = dataset_parameters['test_size']
    temporal_res = dataset_parameters['temporal_res']
    
    model_parameters = config.models
    model_type = model_parameters.pop('model_type')

    if model_type == 'GNN':
        model_parameters['edge_features'] = edge_features
    elif model_type == 'MLP':
        model_parameters.num_nodes = num_nodes

    model = get_model(model_type)(
        node_features=node_features,
        previous_t=previous_t,
        device=device,
        **model_parameters).to(device)

    trainer_options = config.trainer_options
    batch_size = trainer_options.pop('batch_size')

    lr_info = config['lr_info']

    # Model optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr_info['learning_rate'], weight_decay=lr_info['weight_decay'])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_info['step_size'], gamma=lr_info['gamma'])

    # Batch creation
    train_loader = DataLoader(temporal_train_dataset, batch_size=batch_size, shuffle=True)

    # track gradients
    wandb.watch(model, log="all", log_freq=10)

    # info for testing dataset
    temporal_test_dataset_parameters = get_temporal_test_dataset_parameters(
        config, temporal_dataset_parameters)


    # Training
    trainer = Trainer(optimizer, lr_scheduler, **trainer_options)
    trainer.fit(model, train_loader, val_dataset, **temporal_test_dataset_parameters)
    trainer._save_model(model, model_name=f'{wandb.run.id}.h5')

    # Rollout error and time
    spatial_analyser = SpatialAnalysis(model, test_dataset, **temporal_test_dataset_parameters)

    spatial_analyser = SpatialAnalysis(model, test_dataset, **temporal_test_dataset_parameters)

    fig, _ = spatial_analyser.plot_CSI_rollouts(water_thresholds=[0.05, 0.3])
    fig.savefig("results/temp_CSI.png")
    img = cv2.imread("results/temp_CSI.png")
    image = wandb.Image(PIL.Image.fromarray(img), caption="CSI scores")
    wandb.log({"CSI scores": image})

    
    for val_dataset in test_dataset:
        rollout_plotter = PlotRollout(model, val_dataset, scalers=scalers, 
        type_loss=trainer.type_loss, **temporal_test_dataset_parameters)
    # Call the explore_rollout method
    fig_real_WD, fig_pred_WD = rollout_plotter.DEM_with_water()

    # Save or display the figures
    fig_real_WD.savefig("results/real_WD_figure.png")
    fig_pred_WD.savefig("results/predicted_WD_figure.png")
    
    # Log the figures to wandb
    wandb.log({"Real WD Figure": wandb.Image(fig_real_WD)})
    wandb.log({"Predicted WD Figure": wandb.Image(fig_pred_WD)})    

    
    print('Training and testing finished!')

if __name__ == '__main__':
    # Read configuration file with parameters
    cfg = read_config('config_show.yaml')

    wandb.init(
        config=cfg,
    )

    fix_dict_in_config(wandb)

    config = wandb.config

    main(config)