from parlai.scripts.interactive import Interactive

if __name__ == '__main__':
    Interactive.main(
        model_file='zoo:blender/blender_90M/model',
        task='internal'
    )
