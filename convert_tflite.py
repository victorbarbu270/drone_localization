import os

def convert_tflite_to_header():
    tflite_model_path = os.path.join('models', 'motion_classifier.tflite')
    output_path = os.path.join('motion_classifier', 'model.h')
    
    # Read the TFLite model
    with open(tflite_model_path, 'rb') as f:
        model_data = f.read()
    
    # Write the header file
    with open(output_path, 'w') as f:
        f.write('#ifndef MODEL_H_\n')
        f.write('#define MODEL_H_\n\n')
        f.write('// TFLite model data - automatically generated\n')
        f.write('alignas(8) const unsigned char model[] = {\n')
        
        # Write the model data as a hex array
        for i in range(0, len(model_data), 12):
            f.write('    ')
            for j in range(12):
                if i + j < len(model_data):
                    f.write('0x{:02x}, '.format(model_data[i + j]))
            f.write('\n')
        
        f.write('};\n\n')
        f.write('const unsigned int model_len = {:d};\n'.format(len(model_data)))
        f.write('\n#endif  // MODEL_H_\n')
    
    print(f'Model converted and saved to {output_path}')
    print(f'Model size: {len(model_data) / 1024:.2f} KB')

if __name__ == '__main__':
    convert_tflite_to_header() 