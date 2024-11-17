// @ts-nocheck
// Preventing TS checks with files presented in the video for a better presentation.
import type { Message } from 'ai';
import React, { type RefCallback, useEffect } from 'react';
import { ClientOnly } from 'remix-utils/client-only';
import { Menu } from '~/components/sidebar/Menu.client';
import { IconButton } from '~/components/ui/IconButton';
import { Workbench } from '~/components/workbench/Workbench.client';
import { classNames } from '~/utils/classNames';
import { MODEL_LIST, DEFAULT_PROVIDER, PROVIDER_LIST, initializeModelList } from '~/utils/constants';
import { Messages } from './Messages.client';
import { SendButton } from './SendButton.client';
import { useState } from 'react';
import { APIKeyManager } from './APIKeyManager';
import Cookies from 'js-cookie';

import styles from './BaseChat.module.scss';
import type { ProviderInfo } from '~/utils/types';

const EXAMPLE_PROMPTS = [
  { text: '使用 React 和 Tailwind 构建一个待办事项应用' },
  { text: '使用 Astro 创建一个简单的博客' },
  { text: '使用 Material UI 创建一个 cookie 同意表单' },
  { text: '制作一个太空入侵者游戏' },
  { text: '如何让一个 div 居中？' },
];

const providerList = PROVIDER_LIST;

const ModelSelector = ({ model, setModel, provider, setProvider, modelList, providerList }) => {
  return (
    <div className="mb-2 flex gap-2">
      <select
        value={provider?.name}
        onChange={(e) => {
          setProvider(providerList.find((p) => p.name === e.target.value));
          const firstModel = [...modelList].find((m) => m.provider == e.target.value);
          setModel(firstModel ? firstModel.name : '');
        }}
        className="flex-1 p-2 rounded-lg border border-bolt-elements-borderColor bg-bolt-elements-prompt-background text-bolt-elements-textPrimary focus:outline-none focus:ring-2 focus:ring-bolt-elements-focus transition-all"
      >
        {providerList.map((provider) => (
          <option key={provider.name} value={provider.name}>
            {provider.name}
          </option>
        ))}
      </select>
      <select
        key={provider?.name}
        value={model}
        onChange={(e) => setModel(e.target.value)}
        style={{ maxWidth: '70%' }}
        className="flex-1 p-2 rounded-lg border border-bolt-elements-borderColor bg-bolt-elements-prompt-background text-bolt-elements-textPrimary focus:outline-none focus:ring-2 focus:ring-bolt-elements-focus transition-all"
      >
        {[...modelList]
          .filter((e) => e.provider == provider?.name && e.name)
          .map((modelOption) => (
            <option key={modelOption.name} value={modelOption.name}>
              {modelOption.label}
            </option>
          ))}
      </select>
    </div>
  );
};

const TEXTAREA_MIN_HEIGHT = 76;

interface BaseChatProps {
  textareaRef?: React.RefObject<HTMLTextAreaElement> | undefined;
  messageRef?: RefCallback<HTMLDivElement> | undefined;
  scrollRef?: RefCallback<HTMLDivElement> | undefined;
  showChat?: boolean;
  chatStarted?: boolean;
  isStreaming?: boolean;
  messages?: Message[];
  enhancingPrompt?: boolean;
  promptEnhanced?: boolean;
  input?: string;
  model?: string;
  setModel?: (model: string) => void;
  provider?: ProviderInfo;
  setProvider?: (provider: ProviderInfo) => void;
  handleStop?: () => void;
  sendMessage?: (event: React.UIEvent, messageInput?: string) => void;
  handleInputChange?: (event: React.ChangeEvent<HTMLTextAreaElement>) => void;
  enhancePrompt?: () => void;
}

export const BaseChat = React.forwardRef<HTMLDivElement, BaseChatProps>(
  (
    {
      textareaRef,
      messageRef,
      scrollRef,
      showChat = true,
      chatStarted = false,
      isStreaming = false,
      enhancingPrompt = false,
      promptEnhanced = false,
      messages,
      input = '',
      model,
      setModel,
      provider,
      setProvider,
      sendMessage,
      handleInputChange,
      enhancePrompt,
      handleStop,
    },
    ref,
  ) => {
    const TEXTAREA_MAX_HEIGHT = chatStarted ? 400 : 200;
    const [apiKeys, setApiKeys] = useState<Record<string, string>>({});
    const [modelList, setModelList] = useState(MODEL_LIST);

    useEffect(() => {
      // Load API keys from cookies on component mount
      try {
        const storedApiKeys = Cookies.get('apiKeys');
        if (storedApiKeys) {
          const parsedKeys = JSON.parse(storedApiKeys);
          if (typeof parsedKeys === 'object' && parsedKeys !== null) {
            setApiKeys(parsedKeys);
          }
        }
      } catch (error) {
        console.error('Error loading API keys from cookies:', error);
        // Clear invalid cookie data
        Cookies.remove('apiKeys');
      }

      initializeModelList().then((modelList) => {
        setModelList(modelList);
      });
    }, []);

    const updateApiKey = (provider: string, key: string) => {
      try {
        const updatedApiKeys = { ...apiKeys, [provider]: key };
        setApiKeys(updatedApiKeys);
        // Save updated API keys to cookies with 30 day expiry and secure settings
        Cookies.set('apiKeys', JSON.stringify(updatedApiKeys), {
          expires: 30, // 30 days
          secure: true, // Only send over HTTPS
          sameSite: 'strict', // Protect against CSRF
          path: '/', // Accessible across the site
        });
      } catch (error) {
        console.error('Error saving API keys to cookies:', error);
      }
    };

    return (
      <div
        ref={ref}
        className={classNames(
          styles.BaseChat,
          'relative flex h-full w-full overflow-hidden bg-bolt-elements-background-depth-1',
        )}
        data-chat-visible={showChat}
      >
        <ClientOnly>{() => <Menu />}</ClientOnly>
        <div ref={scrollRef} className="flex overflow-y-auto w-full h-full">
          <div className={classNames(styles.Chat, 'flex flex-col flex-grow min-w-[var(--chat-min-width)] h-full')}>
            {!chatStarted && (
              <div id="intro" className="mt-[26vh] max-w-chat mx-auto text-center">
                <h1 className="text-6xl font-bold text-bolt-elements-textPrimary mb-4 animate-fade-in">
                  创意的起点
                </h1>
                <p className="text-xl mb-8 text-bolt-elements-textSecondary animate-fade-in animation-delay-200">
                  在几秒钟内将想法变为现实，或获取现有项目的帮助。
                </p>
              </div>
            )}
            <div
              className={classNames('pt-6 px-6', {
                'h-full flex flex-col': chatStarted,
              })}
            >
              <ClientOnly>
                {() => {
                  return chatStarted ? (
                    <Messages
                      ref={messageRef}
                      className="flex flex-col w-full flex-1 max-w-chat px-4 pb-6 mx-auto z-1"
                      messages={messages}
                      isStreaming={isStreaming}
                    />
                  ) : null;
                }}
              </ClientOnly>
              <div
                className={classNames('relative w-full max-w-chat mx-auto z-prompt', {
                  'sticky bottom-0': chatStarted,
                })}
              >
                <ModelSelector
                  key={provider?.name + ':' + modelList.length}
                  model={model}
                  setModel={setModel}
                  modelList={modelList}
                  provider={provider}
                  setProvider={setProvider}
                  providerList={PROVIDER_LIST}
                />
                {provider && (
                  <APIKeyManager
                    provider={provider}
                    apiKey={apiKeys[provider.name] || ''}
                    setApiKey={(key) => updateApiKey(provider.name, key)}
                  />
                )}
                <div
                  className={classNames(
                    'shadow-lg border border-bolt-elements-borderColor bg-bolt-elements-prompt-background backdrop-filter backdrop-blur-[8px] rounded-lg overflow-hidden transition-all',
                  )}
                >
                  <textarea
                    ref={textareaRef}
                    className={`w-full pl-4 pt-4 pr-16 focus:outline-none focus:ring-2 focus:ring-bolt-elements-focus resize-none text-md text-bolt-elements-textPrimary placeholder-bolt-elements-textTertiary bg-transparent transition-all`}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter') {
                        if (event.shiftKey) {
                          return;
                        }

                        event.preventDefault();

                        sendMessage?.(event);
                      }
                    }}
                    value={input}
                    onChange={(event) => {
                      handleInputChange?.(event);
                    }}
                    style={{
                      minHeight: TEXTAREA_MIN_HEIGHT,
                      maxHeight: TEXTAREA_MAX_HEIGHT,
                    }}
                    placeholder="Bolt 今天能为您做些什么？"
                    translate="no"
                  />
                  <ClientOnly>
                    {() => (
                      <SendButton
                        show={input.length > 0 || isStreaming}
                        isStreaming={isStreaming}
                        onClick={(event) => {
                          if (isStreaming) {
                            handleStop?.();
                            return;
                          }

                          sendMessage?.(event);
                        }}
                      />
                    )}
                  </ClientOnly>
                  <div className="flex justify-between items-center text-sm p-4 pt-2">
                    <div className="flex gap-1 items-center">
                      <IconButton
                        title="Enhance prompt"
                        disabled={input.length === 0 || enhancingPrompt}
                        className={classNames('transition-all', {
                          'opacity-100!': enhancingPrompt,
                          'text-bolt-elements-item-contentAccent! pr-1.5 enabled:hover:bg-bolt-elements-item-backgroundAccent!':
                            promptEnhanced,
                        })}
                        onClick={() => enhancePrompt?.()}
                      >
                        {enhancingPrompt ? (
                          <>
                            <div className="i-svg-spinners:90-ring-with-bg text-bolt-elements-loader-progress text-xl animate-spin"></div>
                            <div className="ml-1.5">正在增强提示...</div>
                          </>
                        ) : (
                          <>
                            <div className="i-bolt:stars text-xl"></div>
                            {promptEnhanced && <div className="ml-1.5">提示已增强</div>}
                          </>
                        )}
                      </IconButton>
                    </div>
                    {input.length > 3 ? (
                      <div className="text-xs text-bolt-elements-textTertiary">
                        按 <kbd className="kdb px-1.5 py-0.5 rounded bg-bolt-elements-background-depth-2">Shift</kbd> +{' '}
                        <kbd className="kdb px-1.5 py-0.5 rounded bg-bolt-elements-background-depth-2">Return</kbd> 换行
                      </div>
                    ) : null}
                  </div>
                </div>
                <div className="bg-bolt-elements-background-depth-1 pb-6">{/* Ghost Element */}</div>
              </div>
            </div>
            {!chatStarted && (
              <div id="examples" className="relative w-full max-w-xl mx-auto mt-8 flex justify-center">
                <div className="flex flex-col space-y-2 [mask-image:linear-gradient(to_bottom,black_0%,transparent_180%)] hover:[mask-image:none]">
                  {EXAMPLE_PROMPTS.map((examplePrompt, index) => {
                    return (
                      <button
                        key={index}
                        onClick={(event) => {
                          sendMessage?.(event, examplePrompt.text);
                        }}
                        className="group flex items-center w-full gap-2 justify-center bg-transparent text-bolt-elements-textTertiary hover:text-bolt-elements-textPrimary transition-theme"
                      >
                        {examplePrompt.text}
                        <div className="i-ph:arrow-bend-down-left" />
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
          <ClientOnly>{() => <Workbench chatStarted={chatStarted} isStreaming={isStreaming} />}</ClientOnly>
        </div>
      </div>
    );
  },
);
