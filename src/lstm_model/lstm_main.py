"""

"""
from lstm_train import train_model, test_model
from lstm_model import LSTMModel, EarlyStopping
from lstm_config import LstmCFG
from lstm_dataset import DemandDataset
from lstm_utils import set_seed, normalize_columns
import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import pandas as pd
import wandb
import os

wandb_config = {
	"n_folds": LstmCFG.n_folds,
	"n_features": LstmCFG.n_features,
	"hidden layers": LstmCFG.hidden_units,
	"learning_rate": LstmCFG.lr,
	"batch_size": LstmCFG.batch_size,
	"epochs": LstmCFG.epochs,
	"sequence_length": LstmCFG.seq_length,
	"dropout": LstmCFG.dropout,
	"num_layers": LstmCFG.num_layers,
	"weight_decay": LstmCFG.weight_decay,
	"lrs_step_size": LstmCFG.lrs_step_size,
	"lrs_gamma": LstmCFG.lrs_gamma,
}

if __name__ == "__main__":
	set_seed(seed_value=42)

	CFG = LstmCFG()

	if CFG.logging:
		wandb.init(
			project=CFG.wandb_project_name,
			name=f'{CFG.wandb_run_name}_v{CFG.version}',
			config=wandb_config,
			job_type='train_model'
		)

	nsw_df = pd.read_parquet(os.path.join(CFG.data_path, 'nsw_df.parquet'))
	nsw_df.drop(
		columns=['daily_avg_actual', 'daily_avg_forecast'],
		inplace=True
	)
	# print(f'nsw_df columns: {nsw_df.shape[1]}') # number of columns in df

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# define a cutoff date from the last date in df
	cutoff_date1 = nsw_df.index.max() - pd.Timedelta(days=7)
	cutoff_date2 = nsw_df.index.max() - pd.Timedelta(days=14)

	# split the data using cutoff date
	train_df = nsw_df[nsw_df.index <= cutoff_date2].copy()
	# print(f'train_df columns: {train_df.shape[1]}')  # dubugging line

	test_df = nsw_df[
		(nsw_df.index > cutoff_date2) & (nsw_df.index <= cutoff_date1)].copy()
	# print(f'nsw_df columns: {test_df.shape[1]}')  # dubugging line

	val_df = nsw_df[nsw_df.index > cutoff_date1].copy()
	# print(f'val_df columns: {val_df.shape[1]}')  # dubugging line

	# normalize the training data, save scaler
	train_df, scalers = normalize_columns(
		train_df,
		LstmCFG.column_mapping
	)

	for fold, (train_index, val_index) in enumerate(tscv.split(train_df)):
		train_sequences = DemandDataset(df=train_df.iloc[train_index],
										label_col=label_col,
										sequence_length=LstmCFG.seq_length)
		test_sequences = DemandDataset(df=train_df.iloc[val_index],
									   label_col=label_col,
									   sequence_length=LstmCFG.seq_length)

		train_loader = DataLoader(train_sequences,
								  batch_size=LstmCFG.batch_size,
								  shuffle=False)
		test_loader = DataLoader(test_sequences,
								 batch_size=LstmCFG.batch_size,
								 shuffle=False)

		model = LSTMModel(
			input_size=LstmCFG.input_size,
			hidden_layer_size=LstmCFG.hidden_units,
			output_size=LstmCFG.output_size,
			dropout=LstmCFG.dropout,
			num_layers=LstmCFG.num_layers
		).to(device)

		optimizer = optim.Adam(
			model.parameters(),
			lr=LstmCFG.lr,
			weight_decay=LstmCFG.weight_decay
		)

		lr_scheduler = torch.optim.lr_scheduler.StepLR(
			optimizer,
			step_size=LstmCFG.lrs_step_size,
			gamma=LstmCFG.lrs_gamma
		)

		loss_function = nn.L1Loss()

		early_stopping = EarlyStopping(
			patience=10,
			verbose=True,
			path='model_checkpoint.pt'
		)

		for epoch in range(LstmCFG.epochs):
			train_loss = train_model(
				model,
				train_loader,
				device,
				optimizer,
				loss_function,
				lr_scheduler,
				CFG)

			test_loss = test_model(
				model,
				test_loader,
				device,
				loss_function
			)

			print(
				"""
				Epoch {epoch + 1}, 
				Train Loss: {train_loss:.4f}, 
				Test Loss: {test_loss:.4f}
				"""
			)

			early_stopping(test_loss, model)

			if early_stopping.early_stop:
				print("Early stopping triggered")
				torch.save(
					model.state_dict(),
					'model_checkpoint.pt'
				)
				break

		model.load_state_dict(torch.load('model_checkpoint.pt'))
		artifact = wandb.Artifact('model_artifact', type='model')
		artifact.add_file('model_checkpoint.pt')
		if CFG.logging:
			wandb.save('model_checkpoint.pt')

		best_train_loss = min(epoch_train_losses)
		all_folds_train_losses.append(best_train_loss)

		best_test_loss = min(epoch_test_losses)
		all_folds_test_losses.append(best_test_loss)

		print(f"Best Train Loss in fold {fold + 1}: {best_train_loss:.4f}")
		print(f"Best Test Loss in fold {fold + 1}: {best_test_loss:.4f}")

		plot_loss_curves(
			epoch_train_losses,
			epoch_test_losses,
			title=f"Fold {fold} Loss Curves"
		)

	model_train_loss = sum(all_folds_train_losses) / len(
		all_folds_train_losses)
	if CFG.logging:
		wandb.log({"model training loss": model_train_loss})
	print(f"Model train loss: {model_train_loss:.4f}")

	model_test_loss = sum(all_folds_test_losses) / len(all_folds_test_losses)
	if CFG.logging:
		wandb.log({"model test loss": model_test_loss})
	print(f"Model test loss: {model_test_loss:.4f}")

	model_gap = model_train_loss - model_test_loss
	if CFG.logging:
		wandb.log({"model gap": model_gap})
		wandb.finish()

	gc.collect()
	torch.cuda.empty_cache()
		# Save and log results as before
