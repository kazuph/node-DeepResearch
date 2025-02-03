import {GoogleGenerativeAI, SchemaType} from "@google/generative-ai";
import {readUrl} from "./tools/read";
import fs from 'fs/promises';
import {SafeSearchType, search as duckSearch} from "duck-duck-scrape";
import {braveSearch} from "./tools/brave-search";
import {rewriteQuery} from "./tools/query-rewriter";
import {dedupQueries} from "./tools/dedup";
import {evaluateAnswer} from "./tools/evaluator";
import {analyzeSteps} from "./tools/error-analyzer";
import {GEMINI_API_KEY, JINA_API_KEY, SEARCH_PROVIDER, STEP_SLEEP, modelConfigs} from "./config";
import {TokenTracker} from "./utils/token-tracker";
import {ActionTracker} from "./utils/action-tracker";
import {StepAction, SchemaProperty, ResponseSchema, AnswerAction} from "./types";
import {TrackerContext} from "./types";

async function sleep(ms: number) {
  const seconds = Math.ceil(ms / 1000);
  console.log(`Waiting ${seconds}s...`);
  return new Promise(resolve => setTimeout(resolve, ms));
}

function getSchema(allowReflect: boolean, allowRead: boolean, allowAnswer: boolean, allowSearch: boolean): ResponseSchema {
  const actions: string[] = [];
  const properties: Record<string, SchemaProperty> = {
    action: {
      type: SchemaType.STRING,
      enum: actions,
      description: "Must match exactly one action type"
    },
    thoughts: {
      type: SchemaType.STRING,
      description: "Explain why choose this action, what's the thought process behind choosing this action"
    }
  };

  if (allowSearch) {
    actions.push("search");
    properties.searchQuery = {
      type: SchemaType.STRING,
      description: "Only required when choosing 'search' action, must be a short, keyword-based query that BM25, tf-idf based search engines can understand."
    };
  }

  if (allowAnswer) {
    actions.push("answer");
    properties.answer = {
      type: SchemaType.STRING,
      description: "Only required when choosing 'answer' action, must be the final answer in natural language"
    };
    properties.references = {
      type: SchemaType.ARRAY,
      items: {
        type: SchemaType.OBJECT,
        properties: {
          exactQuote: {
            type: SchemaType.STRING,
            description: "Exact relevant quote from the document"
          },
          url: {
            type: SchemaType.STRING,
            description: "URL of the document; must be directly from the context"
          }
        },
        required: ["exactQuote", "url"]
      },
      description: "Must be an array of references that support the answer, each reference must contain an exact quote and the URL of the document"
    };
  }

  if (allowReflect) {
    actions.push("reflect");
    properties.questionsToAnswer = {
      type: SchemaType.ARRAY,
      items: {
        type: SchemaType.STRING,
        description: "each question must be a single line, concise and clear. not composite or compound, less than 20 words."
      },
      description: "List of most important questions to fill the knowledge gaps of finding the answer to the original question",
      maxItems: 2
    };
  }

  if (allowRead) {
    actions.push("visit");
    properties.URLTargets = {
      type: SchemaType.ARRAY,
      items: {
        type: SchemaType.STRING
      },
      maxItems: 2,
      description: "Must be an array of URLs, choose up the most relevant 2 URLs to visit"
    };
  }

  // Update the enum values after collecting all actions
  properties.action.enum = actions;

  return {
    type: SchemaType.OBJECT,
    properties,
    required: ["action", "thoughts"]
  };
}

function getPrompt(
  question: string,
  context?: string[],
  allQuestions?: string[],
  allowReflect: boolean = true,
  allowAnswer: boolean = true,
  allowRead: boolean = true,
  allowSearch: boolean = true,
  badContext?: { question: string, answer: string, evaluation: string, recap: string; blame: string; improvement: string; }[],
  knowledge?: { question: string; answer: string; }[],
  allURLs?: Record<string, string>,
  beastMode?: boolean
): string {
  const sections: string[] = [];

  // Add header section
  sections.push(`現在の日時: ${new Date().toUTCString()}

あなたは多段階推論を専門とする高度なAIリサーチアナリストです。トレーニングデータと過去の学習経験を活用して、以下の質問に確実に回答してください：

## 質問
${question}`);

  // Add context section if exists
  if (context?.length) {
    sections.push(`## これまでの行動
以下の行動を実行しました：

${context.join('\n')}`);
  }

  // Add knowledge section if exists
  if (knowledge?.length) {
    const knowledgeItems = knowledge
      .map((k, i) => `### 知識 ${i + 1}: ${k.question}\n${k.answer}`)
      .join('\n\n');

    sections.push(`## 収集した知識
元の質問に回答するのに役立つ可能性のある知識を収集しました。以下が現在までに収集した知識です：

${knowledgeItems}`);
  }

  // Add bad context section if exists
  if (badContext?.length) {
    const attempts = badContext
      .map((c, i) => `### 試行 ${i + 1}
- 質問: ${c.question}
- 回答: ${c.answer}
- 却下理由: ${c.evaluation}
- 行動の要約: ${c.recap}
- 問題点: ${c.blame}`)
      .join('\n\n');

    const learnedStrategy = badContext.map(c => c.improvement).join('\n');

    sections.push(`## 失敗した試行
以下の行動を試みましたが、質問への回答を見つけることができませんでした。

${attempts}

## 学んだ戦略
${learnedStrategy}
`);
  }

  // Build actions section
  const actions: string[] = [];

  if (allURLs && Object.keys(allURLs).length > 0 && allowRead) {
    const urlList = Object.entries(allURLs)
      .map(([url, desc]) => `  + "${url}": "${desc}"`)
      .join('\n');

    actions.push(`**visit**:
- Select the most relevant URLs from the following list to gather external knowledge
${urlList}
- Investigate specific URL content in detail when search results are sufficient
- Full content of URLs is accessible`);
  }

  if (allowSearch) {
    actions.push(`**search**:
- Use public search engines to find external sources
- Focus on specific aspects of the question
- Use keyword-based search queries only, not complete sentences`);
  }

  if (allowAnswer) {
    actions.push(`**answer**:
- Provide final answer only when 100% certain
- Responses must be definitive (no ambiguity, uncertainty, or disclaimers)${allowReflect ? '\n- Use "reflect" if questions remain' : ''}`);
  }

  if (beastMode) {
    actions.push(`**answer**:
- You have gathered enough information to answer the question; they may not be perfect, but this is your very last chance to answer the question.
- Use the most advanced reasoning ability to analyze every detail in the context.
- When uncertain, make educated guesses based on available context and knowledge.
- Responses must be definitive and comprehensive.
- Structure your analysis using MECE (Mutually Exclusive, Collectively Exhaustive) principles.
- Format your answer as a detailed analytical report using this structure:

# Investigation Report

## Executive Summary
[1-2 paragraphs summarizing key findings]

## Key Findings
- Finding 1 [with supporting evidence]
- Finding 2 [with supporting evidence]
- Finding 3 [with supporting evidence]
[etc...]

## Detailed Analysis
### Context Overview
[Background information and context]

### Primary Analysis
[Core analysis of main points]

### Secondary Considerations
[Additional relevant factors]

### Evidence Assessment
- Source 1: [evaluation]
- Source 2: [evaluation]
[etc...]

## Conclusions
[Definitive statements based on analysis]

## Recommendations
- Recommendation 1
- Recommendation 2
[etc...]

## References
- [Source 1]
- [Source 2]
[etc...]

Note: All sections must be comprehensive and backed by evidence from the provided context.`);
  }

  if (allowReflect) {
    actions.push(`**reflect**:
- 仮説的なシナリオや体系的な分析を通じて重要な分析を実行
- 知識のギャップを特定し、必要な明確化のための質問を作成
- 質問の要件:
  - オリジナルであること（既存の質問の変形は不可）
  - 単一の概念に焦点を当てる
  - 20語以内
  - 複合的/複雑な質問は避ける`);
  }

  sections.push(`## 実行可能なアクション

現在のコンテキストに基づいて、以下のアクションの中から1つを選択してください：

${actions.join('\n\n')}`);

  // Add footer
  sections.push(`厳密なJSON形式でのみ応答してください。

重要な要件:
- 1つのアクションタイプのみを含める
- サポートされていないキーは追加しない
- JSON以外のテキスト、マークダウン、説明は含めない
- 厳密なJSON構文を維持する`);

  return sections.join('\n\n');
}

const allContext: StepAction[] = [];  // all steps in the current session, including those leads to wrong results

function updateContext(step: any) {
  allContext.push(step)
}

function removeAllLineBreaks(text: string) {
  return text.replace(/(\r\n|\n|\r)/gm, " ");
}

export async function getResponse(question: string, tokenBudget: number = 1_000_000,
                                  maxBadAttempts: number = 3,
                                  existingContext?: Partial<TrackerContext>): Promise<{ result: StepAction; context: TrackerContext }> {
  const context: TrackerContext = {
    tokenTracker: existingContext?.tokenTracker || new TokenTracker(tokenBudget),
    actionTracker: existingContext?.actionTracker || new ActionTracker()
  };
  context.actionTracker.trackAction({ gaps: [question], totalStep: 0, badAttempts: 0 });
  let step = 0;
  let totalStep = 0;
  let badAttempts = 0;
  const gaps: string[] = [question];  // All questions to be answered including the orginal question
  const allQuestions = [question];
  const allKeywords = [];
  const allKnowledge = [];  // knowledge are intermedidate questions that are answered
  const badContext = [];
  let diaryContext = [];
  let allowAnswer = true;
  let allowSearch = true;
  let allowRead = true;
  let allowReflect = true;
  let prompt = '';
  let thisStep: StepAction = {action: 'answer', answer: '', references: [], thoughts: ''};
  let isAnswered = false;

  const allURLs: Record<string, string> = {};
  const visitedURLs: string[] = [];
  while (context.tokenTracker.getTotalUsage() < tokenBudget && badAttempts <= maxBadAttempts) {
    // add 1s delay to avoid rate limiting
    await sleep(STEP_SLEEP);
    step++;
    totalStep++;
    context.actionTracker.trackAction({ totalStep, thisStep, gaps, badAttempts });
    const budgetPercentage = (context.tokenTracker.getTotalUsage() / tokenBudget * 100).toFixed(2);
    console.log(`Step ${totalStep} / Budget used ${budgetPercentage}%`);
    console.log('Gaps:', gaps);
    allowReflect = allowReflect && (gaps.length <= 1);
    const currentQuestion = gaps.length > 0 ? gaps.shift()! : question;
    // update all urls with buildURLMap
    allowRead = allowRead && (Object.keys(allURLs).length > 0);
    allowSearch = allowSearch && (Object.keys(allURLs).length < 20);  // disable search when too many urls already

    // generate prompt for this step
    prompt = getPrompt(
      currentQuestion,
      diaryContext,
      allQuestions,
      allowReflect,
      allowAnswer,
      allowRead,
      allowSearch,
      badContext,
      allKnowledge,
      allURLs,
      false
      );

    const model = genAI.getGenerativeModel({
      model: modelConfigs.agent.model,
      generationConfig: {
        temperature: modelConfigs.agent.temperature,
        responseMimeType: "application/json",
        responseSchema: getSchema(allowReflect, allowRead, allowAnswer, allowSearch)
      }
    });

    // Check if we have enough budget for this operation (estimate 50 tokens for prompt + response)
    const estimatedTokens = 50;
    const currentUsage = context.tokenTracker.getTotalUsage();
    if (currentUsage + estimatedTokens > tokenBudget) {
      throw new Error(`Token budget would be exceeded: ${currentUsage + estimatedTokens} > ${tokenBudget}`);
    }

    const result = await model.generateContent(prompt);
    const response = await result.response;
    const usage = response.usageMetadata;
    context.tokenTracker.trackUsage('agent', usage?.totalTokenCount || 0);


    thisStep = JSON.parse(response.text());
    // print allowed and chose action
    const actionsStr = [allowSearch, allowRead, allowAnswer, allowReflect].map((a, i) => a ? ['search', 'read', 'answer', 'reflect'][i] : null).filter(a => a).join(', ');
    console.log(`${thisStep.action} <- [${actionsStr}]`);
    console.log(thisStep)

    // reset allowAnswer to true
    allowAnswer = true;
    allowReflect = true;
    allowRead = true;
    allowSearch = true;

    // execute the step and action
    if (thisStep.action === 'answer') {
      updateContext({
        totalStep,
        question: currentQuestion,
        ...thisStep,
      });

      const {response: evaluation} = await evaluateAnswer(currentQuestion, thisStep.answer, context.tokenTracker);


      if (currentQuestion === question) {
        if (badAttempts >= maxBadAttempts) {
          // EXIT POINT OF THE PROGRAM!!!!
          diaryContext.push(`
At step ${step} and ${badAttempts} attempts, you took **answer** action and found an answer, not a perfect one but good enough to answer the original question:

Original question: 
${currentQuestion}

Your answer: 
${thisStep.answer}

The evaluator thinks your answer is good because: 
${evaluation.reasoning}

Your journey ends here.
`);
          isAnswered = false;
          break
        }
        if (evaluation.is_definitive) {
          if (thisStep.references?.length > 0 || Object.keys(allURLs).length === 0) {
            // EXIT POINT OF THE PROGRAM!!!!
            diaryContext.push(`
At step ${step}, you took **answer** action and finally found the answer to the original question:

Original question: 
${currentQuestion}

Your answer: 
${thisStep.answer}

The evaluator thinks your answer is good because: 
${evaluation.reasoning}

Your journey ends here. You have successfully answered the original question. Congratulations! 🎉
`);
            isAnswered = true;
            break
          } else {
            diaryContext.push(`
At step ${step}, you took **answer** action and finally found the answer to the original question:

Original question: 
${currentQuestion}

Your answer: 
${thisStep.answer}

Unfortunately, you did not provide any references to support your answer. 
You need to find more URL references to support your answer.`);
          }

          isAnswered = true;
          break

        } else {
          diaryContext.push(`
At step ${step}, you took **answer** action but evaluator thinks it is not a good answer:

Original question: 
${currentQuestion}

Your answer: 
${thisStep.answer}

The evaluator thinks your answer is bad because: 
${evaluation.reasoning}
`);
          // store the bad context and reset the diary context
          const {response: errorAnalysis} = await analyzeSteps(diaryContext);

          badContext.push({
            question: currentQuestion,
            answer: thisStep.answer,
            evaluation: evaluation.reasoning,
            ...errorAnalysis
          });
          badAttempts++;
          allowAnswer = false;  // disable answer action in the immediate next step
          diaryContext = [];
          step = 0;
        }
      } else if (evaluation.is_definitive) {
        diaryContext.push(`
At step ${step}, you took **answer** action. You found a good answer to the sub-question:

Sub-question: 
${currentQuestion}

Your answer: 
${thisStep.answer}

The evaluator thinks your answer is good because: 
${evaluation.reasoning}

Although you solved a sub-question, you still need to find the answer to the original question. You need to keep going.
`);
        allKnowledge.push({
          question: currentQuestion,
          answer: thisStep.answer,
          type: 'qa'
        });
      }
    } else if (thisStep.action === 'reflect' && thisStep.questionsToAnswer) {
      let newGapQuestions = thisStep.questionsToAnswer
      const oldQuestions = newGapQuestions;
      if (allQuestions.length) {
        newGapQuestions = (await dedupQueries(newGapQuestions, allQuestions)).unique_queries;
      }
      if (newGapQuestions.length > 0) {
        // found new gap questions
        diaryContext.push(`
At step ${step}, you took **reflect** and think about the knowledge gaps. You found some sub-questions are important to the question: "${currentQuestion}"
You realize you need to know the answers to the following sub-questions:
${newGapQuestions.map((q: string) => `- ${q}`).join('\n')}

You will now figure out the answers to these sub-questions and see if they can help me find the answer to the original question.
`);
        gaps.push(...newGapQuestions);
        allQuestions.push(...newGapQuestions);
        gaps.push(question);  // always keep the original question in the gaps
      } else {
        diaryContext.push(`
At step ${step}, you took **reflect** and think about the knowledge gaps. You tried to break down the question "${currentQuestion}" into gap-questions like this: ${oldQuestions.join(', ')} 
But then you realized you have asked them before. You decided to to think out of the box or cut from a completely different angle. 
`);
        updateContext({
          totalStep,
          ...thisStep,
          result: 'I have tried all possible questions and found no useful information. I must think out of the box or different angle!!!'
        });

        allowReflect = false;
      }
    } else if (thisStep.action === 'search' && thisStep.searchQuery) {
      // rewrite queries
      let {queries: keywordsQueries} = await rewriteQuery(thisStep);

      const oldKeywords = keywordsQueries;
      // avoid exisitng searched queries
      if (allKeywords.length) {
        const {unique_queries: dedupedQueries} = await dedupQueries(keywordsQueries, allKeywords);
        keywordsQueries = dedupedQueries;
      }
      if (keywordsQueries.length > 0) {
        const searchResults = [];
        for (const query of keywordsQueries) {
          console.log(`Search query: ${query}`);
          let results;
          if (SEARCH_PROVIDER === 'duck') {
            results = await duckSearch(query, {
              safeSearch: SafeSearchType.STRICT
            });
          } else {
            const {response} = await braveSearch(query);
            await sleep(STEP_SLEEP);
            const searchResults = response.web?.results || response.results || [];
            results = {
              results: searchResults.map((r: any) => ({
                title: r.title,
                url: r.url,
                description: r.description
              }))
            };
          }
          const minResults = results.results.map(r => ({
            title: r.title,
            url: r.url,
            description: r.description,
          }));
          for (const r of minResults) {
            allURLs[r.url] = r.title;
          }
          searchResults.push({query, results: minResults});
          allKeywords.push(query);
        }
        diaryContext.push(`
At step ${step}, you took the **search** action and look for external information for the question: "${currentQuestion}".
In particular, you tried to search for the following keywords: "${keywordsQueries.join(', ')}".
You found quite some information and add them to your URL list and **visit** them later when needed. 
`);

        updateContext({
          totalStep,
          question: currentQuestion,
          ...thisStep,
          result: searchResults
        });
      } else {
        diaryContext.push(`
At step ${step}, you took the **search** action and look for external information for the question: "${currentQuestion}".
In particular, you tried to search for the following keywords: ${oldKeywords.join(', ')}. 
But then you realized you have already searched for these keywords before.
You decided to think out of the box or cut from a completely different angle.
`);


        updateContext({
          totalStep,
          ...thisStep,
          result: 'I have tried all possible queries and found no new information. I must think out of the box or different angle!!!'
        });

        allowSearch = false;
      }
    } else if (thisStep.action === 'visit' && thisStep.URLTargets?.length) {

      let uniqueURLs = thisStep.URLTargets;
      if (visitedURLs.length > 0) {
        // check duplicate urls
        uniqueURLs = uniqueURLs.filter((url: string) => !visitedURLs.includes(url));
      }

      if (uniqueURLs.length > 0) {

        const urlResults = await Promise.all(
          uniqueURLs.map(async (url: string) => {
            const {response, tokens} = await readUrl(url, JINA_API_KEY, context.tokenTracker);
            allKnowledge.push({
              question: `What is in ${response.data?.url || 'the URL'}?`,
              answer: removeAllLineBreaks(response.data?.content || 'No content available'),
              type: 'url'
            });
            visitedURLs.push(url);
            delete allURLs[url];
            return {url, result: response, tokens};
          })
        );
        diaryContext.push(`
At step ${step}, you took the **visit** action and deep dive into the following URLs:
${thisStep.URLTargets.join('\n')}
You found some useful information on the web and add them to your knowledge for future reference.
`);
        updateContext({
          totalStep,
          question: currentQuestion,
          ...thisStep,
          result: urlResults
        });
      } else {

        diaryContext.push(`
At step ${step}, you took the **visit** action and try to visit the following URLs:
${thisStep.URLTargets.join('\n')}
But then you realized you have already visited these URLs and you already know very well about their contents.

You decided to think out of the box or cut from a completely different angle.`);

        updateContext({
          totalStep,
          ...thisStep,
          result: 'I have visited all possible URLs and found no new information. I must think out of the box or different angle!!!'
        });

        allowRead = false;
      }
    }

    await storeContext(prompt, [allContext, allKeywords, allQuestions, allKnowledge], totalStep);
  }
  step++;
  totalStep++;
  await storeContext(prompt, [allContext, allKeywords, allQuestions, allKnowledge], totalStep);
  if (isAnswered) {
    return { result: thisStep, context };
  } else {
    console.log('Enter Beast mode!!!')
    const prompt = getPrompt(
      question,
      diaryContext,
      allQuestions,
      false,
      false,
      false,
      false,
      badContext,
      allKnowledge,
      allURLs,
      true
      );

    const model = genAI.getGenerativeModel({
      model: modelConfigs.agentBeastMode.model,
      generationConfig: {
        temperature: modelConfigs.agentBeastMode.temperature,
        responseMimeType: "application/json",
        responseSchema: getSchema(false, false, allowAnswer, false)
      }
    });

    // Check if we have enough budget for this operation (estimate 50 tokens for prompt + response)
    const estimatedTokens = 50;
    const currentUsage = context.tokenTracker.getTotalUsage();
    if (currentUsage + estimatedTokens > tokenBudget) {
      throw new Error(`Token budget would be exceeded: ${currentUsage + estimatedTokens} > ${tokenBudget}`);
    }

    const result = await model.generateContent(prompt);
    const response = await result.response;
    const usage = response.usageMetadata;
    context.tokenTracker.trackUsage('agent', usage?.totalTokenCount || 0);

    await storeContext(prompt, [allContext, allKeywords, allQuestions, allKnowledge], totalStep);
    thisStep = JSON.parse(response.text());
    console.log(thisStep)
    return { result: thisStep, context };
  }
}

async function storeContext(prompt: string, memory: any[][], step: number) {
  try {
    // Ensure the tmp directory exists
    await fs.mkdir('tmp', { recursive: true });
    await fs.writeFile(`tmp/prompt-${step}.txt`, prompt, 'utf-8');
    const [context, keywords, questions, knowledge] = memory;
    await fs.writeFile('tmp/context.json', JSON.stringify(context, null, 2), 'utf-8');
    await fs.writeFile('tmp/queries.json', JSON.stringify(keywords, null, 2), 'utf-8');
    await fs.writeFile('tmp/questions.json', JSON.stringify(questions, null, 2), 'utf-8');
    await fs.writeFile('tmp/knowledge.json', JSON.stringify(knowledge, null, 2), 'utf-8');
  } catch (error) {
    console.error('Context storage failed:', error);
  }
}

// 新規: タイムスタンプを "YYYY-MM-DD-HHMM" の形式で生成する関数
function getTimestamp(): string {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, '0');
  const day = String(now.getDate()).padStart(2, '0');
  const hours = String(now.getHours()).padStart(2, '0');
  const minutes = String(now.getMinutes()).padStart(2, '0');
  return `${year}-${month}-${day}-${hours}${minutes}`;
}

// 新規: レポートのタイトル（最初の "# " 行）を抽出してファイル名用にフォーマットする関数
function extractTitle(report: string): string {
  const lines = report.split('\n');
  for (const line of lines) {
    if (line.startsWith('# ')) {
      return line.substring(2).trim().replace(/\s+/g, '_');
    }
  }
  return 'report';
}

// 新規: 日本語への翻訳を Gemini 2.0 Flash Exp を利用して実装
async function translateToJapanese(report: string): Promise<string> {
  const translationModel = genAI.getGenerativeModel({
    model: "gemini-2.0-flash-exp",
    generationConfig: {
      temperature: 0,
      responseMimeType: "text/plain"
    }
  });

  const prompt = `Translate the following English text into Japanese:\n\n${report}`;
  const result = await translationModel.generateContent(prompt);
  const response = await result.response;
  return response.text();
}

// 新規: 最終レポートを英語版と日本語訳版として outputs ディレクトリに保存する関数
async function saveFinalReport(report: string): Promise<void> {
  const timestamp = getTimestamp();
  const title = extractTitle(report);
  const englishFileName = `outputs/${timestamp}-${title}.md`;
  const japaneseFileName = `outputs/${timestamp}-${title}-ja.md`;

  // outputs ディレクトリを存在しなければ作成
  await fs.mkdir('outputs', { recursive: true });

  // 英語版レポートの保存
  await fs.writeFile(englishFileName, report, 'utf-8');
  console.log(`English report saved to ${englishFileName}`);

  // 日本語版レポートの生成と保存
  const translatedReport = await translateToJapanese(report);
  await fs.writeFile(japaneseFileName, translatedReport, 'utf-8');
  console.log(`Japanese report saved to ${japaneseFileName}`);
}

const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);


export async function main() {
  const question = process.argv[2] || "";
  const { result: finalStep, context: tracker } = await getResponse(question) as { result: AnswerAction; context: TrackerContext };
  console.log('Final Answer:', finalStep.answer);

  // 新規: 最終レポート（マークダウン形式）を outputs ディレクトリに保存する
  await saveFinalReport(finalStep.answer);

  tracker.tokenTracker.printSummary();
}

if (require.main === module) {
  main().catch(console.error);
}
