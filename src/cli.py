"""Dependency-free argument parsers for public entry points."""
import argparse


def parse_args_training(input_args=None):
    parser = argparse.ArgumentParser(description='Omni-RAD rate-distortion training and GAN fine-tuning')
    parser.add_argument('--stage', type=int, choices=[0, 1, 2], default=1)
    parser.add_argument('--config_dir', default='./configs')
    parser.add_argument('--MASTER_PORT', type=int, default=12355)
    parser.add_argument('--experiment_name', default=None)
    return parser.parse_args(input_args)


def parse_args_testing(input_args=None):
    parser = argparse.ArgumentParser(description='Omni-RAD estimated-rate evaluation')
    parser.add_argument('--stage', type=int, choices=[0, 1, 2], default=1)
    parser.add_argument('--base_config_file', default='./configs/base.yaml')
    parser.add_argument('--test_config_file', default='./configs/test.yaml')
    return parser.parse_args(input_args)
