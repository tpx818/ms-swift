#!/usr/bin/env python3
"""Test script to verify activation_cpu_offload modifications."""

import os
import sys
from unittest.mock import Mock, MagicMock

# Add the swift directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from swift.plugin.activation_cpu_offload import ActivationCpuOffloadCallBack
    from transformers.training_args import TrainingArguments
    print("Successfully imported modules")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def test_on_train_begin_logic():
    """Test the logic of on_train_begin method."""
    print("\nTesting on_train_begin method logic...")
    
    # Create callback instance
    callback = ActivationCpuOffloadCallBack()
    
    # Test 1: Model is not FSDP or FSDP2
    print("Test 1: Model is not FSDP/FSDP2")
    model = Mock()
    args = Mock()
    state = Mock()
    control = Mock()
    
    # This should not call enable_activation_offloading
    try:
        callback.on_train_begin(args, state, control, model=model)
        print("  ✓ No error when model is not FSDP/FSDP2")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 2: Model is FSDP, but no fsdp_config
    print("\nTest 2: Model is FSDP, but no fsdp_config")
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    fsdp_model = Mock(spec=FSDP)
    args = Mock()
    args.fsdp_config = None
    
    try:
        callback.on_train_begin(args, state, control, model=fsdp_model)
        print("  ✓ No error when no fsdp_config")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 3: Model is FSDP, fsdp_config is dict, activation_cpu_offload=False
    print("\nTest 3: Model is FSDP, fsdp_config dict, activation_cpu_offload=False")
    args = Mock()
    args.fsdp_config = {
        'activation_cpu_offload': False,
        'fsdp_version': 2,
        'activation_checkpointing': True
    }
    
    try:
        callback.on_train_begin(args, state, control, model=fsdp_model)
        print("  ✓ No error when activation_cpu_offload=False")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 4: Model is FSDP, fsdp_config is dict, activation_cpu_offload=True, fsdp_version=2
    print("\nTest 4: Model is FSDP, fsdp_config dict, activation_cpu_offload=True, fsdp_version=2")
    args = Mock()
    args.fsdp_config = {
        'activation_cpu_offload': True,
        'fsdp_version': 2,
        'activation_checkpointing': True
    }
    
    # Mock the enable_activation_offloading function
    import swift.plugin.activation_cpu_offload as module
    original_enable = module.enable_activation_offloading
    mock_enable = Mock()
    module.enable_activation_offloading = mock_enable
    
    try:
        callback.on_train_begin(args, state, control, model=fsdp_model)
        if mock_enable.called:
            print("  ✓ enable_activation_offloading was called")
            call_args = mock_enable.call_args
            print(f"    - strategy: {call_args[1]['strategy']}")
            print(f"    - enable_ckpt: {call_args[1]['enable_ckpt']}")
        else:
            print("  ✗ enable_activation_offloading was NOT called")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    finally:
        module.enable_activation_offloading = original_enable
    
    # Test 5: Model is FSDP, fsdp_config is dict, activation_cpu_offload=True, fsdp_version=1
    print("\nTest 5: Model is FSDP, fsdp_config dict, activation_cpu_offload=True, fsdp_version=1")
    args = Mock()
    args.fsdp_config = {
        'activation_cpu_offload': True,
        'fsdp_version': 1,
        'activation_checkpointing': False
    }
    
    module.enable_activation_offloading = mock_enable
    mock_enable.reset_mock()
    
    try:
        callback.on_train_begin(args, state, control, model=fsdp_model)
        if mock_enable.called:
            print("  ✓ enable_activation_offloading was called")
            call_args = mock_enable.call_args
            print(f"    - strategy: {call_args[1]['strategy']}")
            print(f"    - enable_ckpt: {call_args[1]['enable_ckpt']}")
        else:
            print("  ✗ enable_activation_offloading was NOT called")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    finally:
        module.enable_activation_offloading = original_enable
    
    # Test 6: Model is FSDP2
    print("\nTest 6: Model is FSDP2")
    from torch.distributed.fsdp import FSDPModule as FSDP2
    fsdp2_model = Mock(spec=FSDP2)
    args = Mock()
    args.fsdp_config = {
        'activation_cpu_offload': True,
        'fsdp_version': 2,
        'activation_checkpointing': True
    }
    
    module.enable_activation_offloading = mock_enable
    mock_enable.reset_mock()
    
    try:
        callback.on_train_begin(args, state, control, model=fsdp2_model)
        if mock_enable.called:
            print("  ✓ enable_activation_offloading was called for FSDP2")
        else:
            print("  ✗ enable_activation_offloading was NOT called for FSDP2")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    finally:
        module.enable_activation_offloading = original_enable
    
    print("\n" + "="*50)
    print("All tests completed!")


if __name__ == "__main__":
    test_on_train_begin_logic()
