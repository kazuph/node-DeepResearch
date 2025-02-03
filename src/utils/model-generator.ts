import { GoogleGenerativeAI, GenerativeModel } from "@google/generative-ai";
import { GEMINI_API_KEY } from "../config";

export async function createModel(config: ModelConfig, schema: any): Promise<GenerativeModel> {
  const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
  
  try {
    const model = genAI.getGenerativeModel({
      model: config.model,
      generationConfig: {
        temperature: config.temperature,
        responseMimeType: "application/json",
        responseSchema: schema
      }
    });
    
    // テスト生成を試みてモデルが利用可能か確認
    await model.generateContent("test");
    return model;
    
  } catch (error: any) {
    if (error.message?.includes('too many') || error.message?.includes('not available')) {
      console.log(`Failed to use ${config.model}, falling back to ${config.fallbackModel}`);
      return genAI.getGenerativeModel({
        model: config.fallbackModel,
        generationConfig: {
          temperature: config.temperature,
          responseMimeType: "application/json",
          responseSchema: schema
        }
      });
    }
    throw error;
  }
} 