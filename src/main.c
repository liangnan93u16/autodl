#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include "models/applio/scripts.h"
#include "models/fish_speech/scripts.h"
#include "models/e2-f5-tts/scripts.h"
#include "models/instantir/scripts.h"
#include "models/bolt/scripts.h"
#include "models/allegrotxt2vid/scripts.h"
#include "models/omnigen/scripts.h"
#include "models/diamond/scripts.h"
#include "models/facepoke/scripts.h"
#include "models/invoke/scripts.h"
#include "models/diffusersimagefill/scripts.h"
#include "models/facefusion/scripts.h"

// 创建临时脚本文件
char* create_temp_script(const char* content) {
    static char temp_path[1024];
    snprintf(temp_path, sizeof(temp_path), "/tmp/auto_dl_script_XXXXXX");
    
    int fd = mkstemp(temp_path);
    if (fd == -1) {
        printf("创建临时文件失败: %s\n", strerror(errno));
        return NULL;
    }
    
    write(fd, content, strlen(content));
    close(fd);
    chmod(temp_path, 0755);
    
    return temp_path;
}

int main() {
    int choice, action;
    
    printf("注意：仅限于学术用途，不承诺稳定性保证\n\n");
    printf("请选择要使用的模型：\n");
    printf("1. Applio（中文界面）\n");
    printf("2. Fish-Speech（中文界面）\n");
    printf("3. E2-F5-TTS\n");
    printf("4. InstantIR\n");
    printf("5. Bolt\n");
    printf("6. AllegroTxt2vid\n");
    printf("7. OmniGen\n");
    printf("8. Diamond\n");
    printf("9. FacePoke\n");
    printf("10. InvokeAI\n");
    printf("11. DiffusersImageFill（中文界面）\n");
    printf("12. FaceFusion（中文界面）\n");
    printf("请输入选项 (1-12): ");
    
    if (scanf("%d", &choice) != 1) {
        printf("输入无效\n");
        return 0;
    }
    
    if (choice < 1 || choice > 12) {
        printf("无效的模型选项\n");
        return 0;
    }
    
    while (getchar() != '\n');
    
    printf("\n请选择操作：\n");
    printf("1. 安装\n");
    printf("2. 启动服务\n");
    printf("请输入选项 (1-2): ");
    
    if (scanf("%d", &action) != 1) {
        printf("输入无效\n");
        return 0;
    }
    
    if (action < 1 || action > 2) {
        printf("无效的操作选项\n");
        return 0;
    }
    
    char command[1024];
    const char* script_content = NULL;
    
    switch(choice) {
        case 1: // Applio
            if (action == 1) {
                script_content = APPLIO_INSTALL;
            } else if (action == 2) {
                script_content = APPLIO_START;
            }
            break;
            
        case 2: // Fish-Speech
            if (action == 1) {
                script_content = FISH_SPEECH_INSTALL;
            } else if (action == 2) {
                script_content = FISH_SPEECH_START;
            }
            break;
            
        case 3: // E2-F5-TTS
            if (action == 1) {
                script_content = E2_F5_TTS_INSTALL;
            } else if (action == 2) {
                script_content = E2_F5_TTS_START;
            }
            break;
            
        case 4: // InstantIR
            if (action == 1) {
                script_content = INSTANTIR_INSTALL;
            } else if (action == 2) {
                script_content = INSTANTIR_START;
            }
            break;
            
        case 5: // Bolt
            if (action == 1) {
                script_content = BOLT_INSTALL;
            } else if (action == 2) {
                script_content = BOLT_START;
            }
            break;
            
        case 6: // AllegroTxt2vid
            if (action == 1) {
                script_content = ALLEGROTXT2VID_INSTALL;
            } else if (action == 2) {
                script_content = ALLEGROTXT2VID_START;
            }
            break;
            
        case 7: // OmniGen
            if (action == 1) {
                script_content = OMNIGEN_INSTALL;
            } else if (action == 2) {
                script_content = OMNIGEN_START;
            }
            break;
            
        case 8: // Diamond
            if (action == 1) {
                script_content = DIAMOND_INSTALL;
            } else if (action == 2) {
                script_content = DIAMOND_START;
            }
            break;
            
        case 9: // FacePoke
            if (action == 1) {
                script_content = FACEPOKE_INSTALL;
            } else if (action == 2) {
                script_content = FACEPOKE_START;
            }
            break;
            
        case 10: // InvokeAI
            if (action == 1) {
                script_content = INVOKE_INSTALL;
            } else if (action == 2) {
                script_content = INVOKE_START;
            }
            break;
            
        case 11: // DiffusersImageFill
            if (action == 1) {
                script_content = DIFFUSERSIMAGEFILL_INSTALL;
            } else if (action == 2) {
                script_content = DIFFUSERSIMAGEFILL_START;
            }
            break;
            
        case 12: // FaceFusion
            if (action == 1) {
                script_content = FACEFUSION_INSTALL;
            } else if (action == 2) {
                script_content = FACEFUSION_START;
            }
            break;
    }
    
    char* temp_script = create_temp_script(script_content);
    if (!temp_script) {
        return 1;
    }
    
    snprintf(command, sizeof(command), "bash '%s'", temp_script);
    int result = system(command);
    
    unlink(temp_script);
    
    if (result == -1) {
        printf("命令执行失败\n");
        return 1;
    }
    
    return 0;
}
