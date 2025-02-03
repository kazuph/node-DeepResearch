import { GoogleGenerativeAI, SchemaType } from "@google/generative-ai";
import { createModel } from '../utils/model-generator';
import { GEMINI_API_KEY, modelConfigs } from "../config";
import { TokenTracker } from "../utils/token-tracker";

import { EvaluationResponse } from '../types';

const responseSchema = {
  type: SchemaType.OBJECT,
  properties: {
    is_definitive: {
      type: SchemaType.BOOLEAN,
      description: "Whether the answer provides a definitive response without uncertainty or 'I don't know' type statements"
    },
    reasoning: {
      type: SchemaType.STRING,
      description: "Detailed explanation using MECE analysis and structured markdown format"
    }
  },
  required: ["is_definitive", "reasoning"]
};

let model: any = null;

function getPrompt(question: string, answer: string): string {
  return `You are an expert evaluator specializing in analyzing the definitiveness and completeness of answers. Analyze whether the given answer provides a definitive response.

Core Evaluation Criteria:
1. Definitiveness:
   - Must identify any uncertainty markers like "I think", "maybe", "probably"
   - Must flag any "I don't know" or "information not available" statements
   - Must detect any hedging or ambiguous language
   
2. Analysis Requirements:
   - Use MECE (Mutually Exclusive, Collectively Exhaustive) principles
   - Break down analysis into clear categories
   - Consider all relevant aspects without overlap
   - Provide comprehensive but structured evaluation
   
3. Output Format:
   - Use markdown for structured presentation
   - Include clear section headers
   - Use bullet points for detailed breakdowns
   - Maintain professional analytical tone

Examples:

Question: "What are the system requirements for Python 3.9?"
Answer: "I'm not entirely sure, but I think you need a computer with RAM."
Evaluation: {
  "is_definitive": false,
  "reasoning": "# Answer Analysis\n\n## Uncertainty Indicators\n- Contains phrase 'not entirely sure'\n- Uses tentative language 'I think'\n\n## Content Assessment\n- Provides vague, non-specific requirements\n- Lacks concrete technical specifications\n\n## Conclusion\nThe answer fails to provide definitive information due to explicit uncertainty and lack of specific details."
}

Question: "What are the system requirements for Python 3.9?"
Answer: "Python 3.9 requires Windows 7 or later, macOS 10.11 or later, or Linux."
Evaluation: {
  "is_definitive": true,
  "reasoning": "# Answer Analysis\n\n## Certainty Indicators\n- Uses clear, declarative statements\n- No hedging or uncertainty markers\n\n## Content Assessment\n- Specifies exact OS versions\n- Covers all major platforms\n- Provides concrete requirements\n\n## Conclusion\nThe answer is definitive, providing clear and specific system requirements without ambiguity."
}

Question: "What is the Twitter account of Jina AI's founder?"
Answer: "The provided text does not contain information about Jina AI founder's Twitter account."
Evaluation: {
  "is_definitive": false,
  "reasoning": "# Answer Analysis\n\n## Response Type\n- Indicates information absence\n- States explicit knowledge gap\n\n## Content Assessment\n- No actual answer provided\n- Acknowledges information limitation\n\n## Conclusion\nThe response is non-definitive as it explicitly states an inability to provide the requested information."
}

Now, evaluate this combination:
Question: ${JSON.stringify(question)}
Answer: ${JSON.stringify(answer)}`;
}

export async function evaluateAnswer(question: string, answer: string, tracker: TokenTracker): Promise<{ response: EvaluationResponse }> {
  if (!model) {
    model = await createModel(modelConfigs.evaluator, responseSchema);
  }
  
  try {
    const result = await model.generateContent(getPrompt(question, answer));
    const response = await result.response;
    const usage = response.usageMetadata;
    tracker.trackUsage('evaluator', usage?.totalTokenCount || 0);
    
    return {
      response: JSON.parse(response.text())
    };
  } catch (error) {
    console.error('Evaluation error:', error);
    // モデルをリセットして次回新しく作成
    model = null;
    throw error;
  }
}

// Example usage
async function main() {
  const question = process.argv[2] || '';
  const answer = process.argv[3] || '';

  if (!question || !answer) {
    console.error('Please provide both question and answer as command line arguments');
    process.exit(1);
  }

  try {
    await evaluateAnswer(question, answer);
  } catch (error) {
    console.error('Failed to evaluate answer:', error);
  }
}

if (require.main === module) {
  main().catch(console.error);
}
