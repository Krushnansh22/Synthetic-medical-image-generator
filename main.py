import argparse
import utils.promptbased.Results as PBResults
import utils.promptbased.train as PB
from utils.preprocessing import preprocess
from utils.DisplayImage import display_image
from utils.promptbased.PromptBuilder import prompt_engineering
from utils.Analysis import compare_embeddings


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode', type=str, required=True,choices=['train','load_pretrained'])
    parser.add_argument('-t', '--type', type=str, choices=['Prompt-Based', 'Direct-Conditional'], help='choose which type of fine-tuning to do: Prompt-Based or Direct-Conditional')
    parser.add_argument('--plot', action='store_true', help='Plot the data distribution info')
    parser.add_argument('--img', action='store_true', help="Display random image from the dataset")
    parser.add_argument('--resolution', default=512, type=int, help='The desired image dimension')
    parser.add_argument('--learning_rate', type=str, help='learning rate for fine-tuning')
    parser.add_argument('--epochs', default=1, type=int, help='Training epochs')
    parser.add_argument('--output', type=str, help='output directory for training weights')
    parser.add_argument('--LoRA_weights', type=str, help='path to LoRA weights')
    parser.add_argument('--prompt', type=str, help='enter prompt used for generation', default='Dermoscopy image of a melanoma on the upper back of a 65-year-old male.')
    parser.add_argument('--num_inf_steps', type=int, default=30, help='number of inference steps during generations')
    parser.add_argument('--guidance_scale', type=int, default=7.5, help='guidance scale for prompt generation')
    parser.add_argument('--analyze_results', action="store_true", help="analyze embeddings of the generated images")
    parser.add_argument('--generated_directory', default='./results/', help='path to generated image directory')
    parser.add_argument('--simulate_queuing', action='store_true', help='simulate generation of images to calculate average time')
    
    args = parser.parse_args()
    
    if args.img:
            display_image(data)
    
    if args.mode == 'train':
        data = preprocess(args.plot)
    
        if args.type == 'Prompt-Based':
            missing = [arg for arg in ['resolution', 'learning_rate', 'epochs', 'output'] if getattr(args, arg) is None]
            if missing:
                parser.error(f"When --mode=train, you must specify: {', '.join('--' + m for m in missing)}")
            
            prompt_engineering(data)
            PB.create_unified_file(data)
            PB.train(args.resolution, args.learning_rate, args.epochs, args.output)
            print('LoRA weights saved')
            
        elif args.type == 'Direct-Conditional':
            pass
        
    elif args.mode == 'load_pretrained':
        if args.type == 'Prompt-Based':
            if not args.LoRA_weights:
                parser.error(f"When --mode=load_pretrained, you must specify LoRA weights path")
            if args.simulate_queuing:
                times = PBResults.sample_finetuned(args.prompt, args.LoRA_weights, args.num_inf_steps, args.guidance_scale, generate=200)
                
            image_base = PBResults.sample_base(args.prompt, args.num_inf_steps, args.guidance_scale)
            image_finetuned = PBResults.sample_finetuned(args.prompt, args.LoRA_weights, args.num_inf_steps, args.guidance_scale)
            PBResults.display_results(image_base, image_finetuned)
            
            if args.analyze_results:
                compare_embeddings(args.generated_directory + "prompt based/finetuned_model.png", args.prompt)
                
        


if __name__ == '__main__':
    main()