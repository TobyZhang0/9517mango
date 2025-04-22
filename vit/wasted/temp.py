
for st in strategies:
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # 训练模型
    model = create_model(num_classes)
    model = model.to(device)
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'epoch_time': [],
        'batch_losses': []  # 保留这个记录，用于训练后的可视化
    }

    best_val_acc = 0.0
    batch_losses = []
    total_batch = 0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 记录当前batch的损失
            batch_losses.append(loss.item())
            total_batch += 1

            # 显示简单的进度条
            if (batch_idx + 1) % 10 == 0 or batch_idx == len(train_loader) - 1:
                batch_acc = (predicted == labels).sum().item() / labels.size(0)
                print(f"Batch进度: {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, 准确率: {batch_acc:.4f}",
                      end='\r')

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct / total

        # 保存每个epoch结束时的所有batch损失
        history['batch_losses'].extend(batch_losses[-len(train_loader):])

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        # 学习率调整
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        # 计算epoch时间
        epoch_time = time.time() - start_time

        # 保存最佳模型
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_vit_model.pth')
            best_mark = "✓ [最佳]"
        else:
            best_mark = ""

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)

        # 美化打印输出
        print(f"\n{'-' * 80}")
        print(f"Epoch {epoch + 1}/{num_epochs} 完成 - 耗时: {epoch_time:.2f}秒 {best_mark}")
        print(f"学习率: {current_lr:.8f} {'→ ' + str(new_lr) if current_lr != new_lr else ''}")
        print(f"训练集 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f} ({correct}/{total})")
        print(f"验证集 - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        if is_best:
            print(f"✓ 新的最佳模型已保存! (验证准确率: {val_acc:.4f})")
        print(f"{'-' * 80}")

    print(f"\n{'-' * 80}")
    print(f"训练完成! 最佳验证准确率: {best_val_acc:.4f}")
    print(f"{'-' * 80}")

