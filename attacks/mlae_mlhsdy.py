import numpy as np
import logging
import torch

from PIL import Image

import gc
from multiprocessing import Pool
from ml_liw_model.train  import criterion
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
class MLDE_R(object):
    def __init__(self, model):
        self.model = model

    def generate_np(self, x_list, **kwargs):
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        logging.info('prepare attack')
        self.clip_max = kwargs['clip_max']
        self.clip_min = kwargs['clip_min']
        y_target = kwargs['y_target']
        eps = kwargs['eps']
        pop_size = kwargs['pop_size']
        generation = kwargs['generation']
        batch_size = kwargs['batch_size']
        x_adv = []
        norm2 = []
        norm1 = []
        maxr = []
        meanr = []
        success = 0
        nchannels,img_rows, img_cols,  = x_list.shape[1:4]
        count = 0
        for i in range(len(x_list)):

                print(y_target[i])
                target_label = np.argwhere(y_target[i] > 0)
                r, count_tem, fit, x_adv1 , fs = MLHS_Dy(pop_size, generation, img_rows * img_cols * nchannels, self.model, x_list[i],
                                       target_label, eps, batch_size, gradient=None)
                print('fit:', fit, '\n')
                print('this is DE final l2 norm:', np.linalg.norm((x_adv1 - x_list[i]).flatten(), ord=2))

                if fs == 1:
                # successfully generate AE
                    print('Reduce R')
                    reduce_g = generation - (count_tem // pop_size)
                    r1, count_tem_r, fit_r, x_adv1 = DE_jianR(pop_size, reduce_g, img_rows * img_cols * nchannels, self.model,
                                                x_list[i],x_adv1,
                                                target_label, eps, batch_size, gradient=None)
                x_adv_tem = x_adv1

                with torch.no_grad():
                    if torch.cuda.is_available():
                        adv_pred = self.model(
                            torch.tensor(np.expand_dims(x_adv_tem, axis=0), dtype=torch.float32).cuda()).cpu()
                    else:
                        adv_pred = self.model(torch.tensor(np.expand_dims(x_adv_tem, axis=0), dtype=torch.float32))
                adv_pred = np.asarray(adv_pred)
                pred = adv_pred.copy()
                pred[pred >= (0.5 + 0)] = 1
                pred[pred < (0.5 + 0)] = -1
                adv_pred_match_target = np.all((pred == y_target[i]), axis=1)
                print('adv_pred_match_target',adv_pred_match_target)
                if fs == 1:
                    success = success + 1
                    count += (count_tem+count_tem_r)
                    norm1.append(np.linalg.norm((x_adv_tem - x_list[i]).flatten(), ord=1))
                    norm2.append(np.linalg.norm((x_adv_tem - x_list[i]).flatten(), ord=2))
                    maxr.append(np.max(np.abs(x_adv_tem - x_list[i]), axis=(0,1, 2)))
                    meanr.append(np.mean(np.abs(x_adv_tem - x_list[i]), axis=(0,1, 2)))
                x_adv.append(x_adv_tem)

                logging.info('Successfully generated adversarial examples on '+str(success)+' of '+str(batch_size)+' instances')
        print('batch count',count)
        print(np.mean(norm2),np.mean(norm1),np.mean(maxr),np.mean(meanr) )
        return x_adv , count, success,np.mean(norm2),np.mean(norm1),np.mean(maxr),np.mean(meanr),np.mean(norm2)/6.0859

class Problem:
    def __init__(self, model, image, target_label, eps, batch_size):
        self.model = model
        self.image = image
        self.target_label = target_label
        self.eps = eps
        self.batch_size = batch_size

    def evaluate(self, x):
        with torch.no_grad():
            if torch.cuda.is_available():
                predict = self.model(torch.tensor(np.clip(np.tile(self.image, (len(x), 1, 1, 1))
                                                          + np.reshape(x, (len(x),) + self.image.shape) * self.eps, 0.,
                                                          1.), dtype=torch.float32).cuda()).cpu()
            else:
                predict = self.model(torch.tensor(np.clip(np.tile(self.image, (len(x), 1, 1, 1))
                                                          + np.reshape(x, (len(x),) + self.image.shape) * self.eps, 0.,
                                                          1.), dtype=torch.float32))
        p = np.copy(predict)
        q = np.zeros(p.shape)+0.5
        fit = p-q
        fit[:,self.target_label]=-fit[:,self.target_label]
        fit[np.where(fit<0)]=0
        fitness = np.sum(fit,axis=1)
        fitness = fitness[:, np.newaxis]
        return fitness, fit

class Problem1:
    def __init__(self, model, image, target_label, eps, batch_size):
        self.model = model
        self.image = image
        self.target_label = target_label
        self.eps = eps
        self.batch_size = batch_size
        # print('self.image norm2',np.linalg.norm(self.image.flatten(), ord=2))
        # print(self.image.shape)

    def evaluate(self, x):
        pop_fitness = []
        pop_fit = []
        
        for i in range(len(x)):
            pic_xi = np.clip(self.image + np.reshape(x[i],  self.image.shape), 0., 1.)
            with torch.no_grad():
                if torch.cuda.is_available():

                    predict_x = self.model(torch.tensor(pic_xi, dtype=torch.float32).unsqueeze(0).cuda()).cpu()

                else:
                    predict_x = self.model(torch.tensor(pic_xi, dtype=torch.float32).unsqueeze(0))

            px = np.copy(predict_x)
            qx = np.zeros(px.shape) + 0.5
            fitx = px - qx
            fitx[:, self.target_label] = -fitx[:, self.target_label]
            fitx[np.where(fitx < 0)] = 0

            pop_fit.append(np.squeeze(fitx))

            fitnessx = np.sum(fitx, axis=1)
            pop_fitness.append(fitnessx)
            # print(fitnessx)
        pop_fitness = np.asarray(pop_fitness)
        pop_fit = np.asarray(pop_fit)

        return pop_fitness, pop_fit
        # return fitness, fit

class Problem2:
    def __init__(self, model,ori, adv, target_label, eps, batch_size):
        self.model = model
        # ԭͼ
        self.ori = ori
        # ��ǰ�Կ�����
        self.adv = adv

        self.target_label = target_label
        self.eps = eps
        self.batch_size = batch_size

    # ԭʼͼ��������
    def evaluate_ini(self, x):
        with torch.no_grad():
            if torch.cuda.is_available():
                predict = self.model(torch.tensor(np.clip(np.tile(self.ori, (len(x), 1, 1, 1))
                                                          + np.reshape(x, (len(x),) + self.ori.shape) , 0.,
                                                          1.), dtype=torch.float32).cuda()).cpu()
            else:
                predict = self.model(torch.tensor(np.clip(np.tile(self.ori, (len(x), 1, 1, 1))
                                                          + np.reshape(x, (len(x),) + self.ori.shape) , 0.,
                                                          1.), dtype=torch.float32))
        p = np.copy(predict)
        q = np.zeros(p.shape)+0.5
        fit = p-q
        fit[:,self.target_label]=-fit[:,self.target_label]
        fit[np.where(fit<0)]=0
        fitness = np.sum(fit,axis=1)
        fitness = fitness[:, np.newaxis]
        return fitness, fit


    def evaluate_mid(self, x):
        with torch.no_grad():
            if torch.cuda.is_available():
                predict = self.model(torch.tensor(np.clip(np.tile(self.adv, (len(x), 1, 1, 1))
                                                          + np.reshape(x, (len(x),) + self.ori.shape) , 0.,
                                                          1.), dtype=torch.float32).cuda()).cpu()
            else:
                predict = self.model(torch.tensor(np.clip(np.tile(self.adv, (len(x), 1, 1, 1))
                                                          + np.reshape(x, (len(x),) + self.ori.shape) , 0.,
                                                          1.), dtype=torch.float32))
        p = np.copy(predict)
        q = np.zeros(p.shape)+0.5
        fit = p-q
        fit[:,self.target_label]=-fit[:,self.target_label]
        fit[np.where(fit<0)]=0
        fitness = np.sum(fit,axis=1)
        fitness = fitness[:, np.newaxis]
        return fitness, fit

    # ����һ������
    def evaluate_one(self, x_adv):
        with torch.no_grad():
            if torch.cuda.is_available():
                predict = self.model(torch.tensor(x_adv, dtype=torch.float32).unsqueeze(0).cuda()).cpu()
            else:
                predict = self.model(torch.tensor(x_adv, dtype=torch.float32).unsqueeze(0))
        p = np.copy(predict)
        q = np.zeros(p.shape)+0.5
        fit = p-q
        fit[:,self.target_label]=-fit[:,self.target_label]
        fit[np.where(fit<0)]=0
        fitness = np.sum(fit,axis=1)
        fitness = fitness[:, np.newaxis]
        l2_n = (np.reshape(x_adv, self.ori.shape)- self.ori).flatten()
        print('evaluate_one',fitness,'l2 norm:',np.linalg.norm(l2_n, ord=2))

    # �õ�ǰ�Կ�����������
    def evaluate_adv(self, x):
        with torch.no_grad():
            if torch.cuda.is_available():
                predict = self.model(torch.tensor(np.clip(np.tile(self.adv, (len(x), 1, 1, 1))
                                                          + np.reshape(x, (len(x),) + self.adv.shape) , 0.,
                                                          1.), dtype=torch.float32).cuda()).cpu()
            else:
                predict = self.model(torch.tensor(np.clip(np.tile(self.adv, (len(x), 1, 1, 1))
                                                          + np.reshape(x, (len(x),) + self.adv.shape) , 0.,
                                                          1.), dtype=torch.float32))
        p = np.copy(predict)
        q = np.zeros(p.shape)+0.5
        fit = p-q
        fit[:,self.target_label]=-fit[:,self.target_label]
        fit[np.where(fit<0)]=0
        fitness = np.sum(fit,axis=1)
        fitness = fitness[:, np.newaxis]

        fitness[np.where(fitness != 0)] = 999

        zero_fitness_indices = np.where(fitness == 0)[0]
        for index in zero_fitness_indices:
            r = (np.reshape(x[index], self.adv.shape) + self.adv - self.ori).flatten()
            fitness[index] = np.linalg.norm(r, ord=2)

        return fitness, fit

class Problem_DE_jianR:
    def __init__(self, model,ori, adv, target_label, eps, batch_size):
        self.model = model
        self.adv = adv
        self.ori = ori
        self.target_label = target_label
        self.eps = eps
        self.batch_size = batch_size


    def evaluate(self, x):
        with torch.no_grad():
            if torch.cuda.is_available():
                predict = self.model(torch.tensor(np.clip(np.tile(self.ori, (len(x), 1, 1, 1))
                                                          + np.reshape(x, (len(x),) + self.ori.shape) , 0.,
                                                          1.), dtype=torch.float32).cuda()).cpu()
            else:
                predict = self.model(torch.tensor(np.clip(np.tile(self.ori, (len(x), 1, 1, 1))
                                                          + np.reshape(x, (len(x),) + self.ori.shape) , 0.,
                                                          1.), dtype=torch.float32))
        p = np.copy(predict)
        q = np.zeros(p.shape)+0.5
        fit = p-q
        fit[:,self.target_label]=-fit[:,self.target_label]
        fit[np.where(fit<0)]=0
        fitness = np.sum(fit,axis=1)
        fitness = fitness[:, np.newaxis]
        return fitness, fit


    def evaluate_ori(self, x):

        neg_r_direction = (self.ori - self.adv).flatten()
        x = np.clip(x,0,1) * neg_r_direction



        with torch.no_grad():
            if torch.cuda.is_available():
                predict = self.model(torch.tensor(np.clip(np.tile(self.adv, (len(x), 1, 1, 1))
                                                          + np.reshape(x, (len(x),) + self.adv.shape) , 0.,
                                                          1.), dtype=torch.float32).cuda()).cpu()
            else:
                predict = self.model(torch.tensor(np.clip(np.tile(self.adv, (len(x), 1, 1, 1))
                                                          + np.reshape(x, (len(x),) + self.adv.shape) , 0.,
                                                          1.), dtype=torch.float32))
        p = np.copy(predict)
        q = np.zeros(p.shape)+0.5
        fit = p-q
        fit[:,self.target_label]=-fit[:,self.target_label]
        fit[np.where(fit<0)]=0
        fitness = np.sum(fit,axis=1)
        fitness = fitness[:, np.newaxis]

        print('ori',fitness)

        for i in range(len(x)):
            r = (np.reshape(x[i], self.adv.shape) + self.adv - self.ori).flatten()
            print(i,np.linalg.norm(r, ord=2))
        # indices = np.where((fitness < 0.1) & (fitness != 0))[0]

        fitness[np.where((fitness >= 0.1) & (fitness != 0))] = 999
        print('change ',fitness)

        for index in np.where((fitness < 0.1) & (fitness != 0))[0]:
            r = (np.reshape(x[index], self.adv.shape) + self.adv - self.ori).flatten()
            print('fitness[index]',fitness[index],np.linalg.norm(r, ord=2) / np.linalg.norm(self.adv.flatten(), ord=2))
            fitness[index] = np.linalg.norm(r, ord=2) / np.linalg.norm(self.adv.flatten(), ord=2) + 10 * fitness[index]
            print('final fitness[index]', fitness[index])

        zero_fitness_indices = np.where(fitness == 0)[0]
        for index in zero_fitness_indices:
            r = (np.reshape(x[index], self.adv.shape) + self.adv - self.ori).flatten()
            fitness[index] = np.linalg.norm(r, ord=2) / np.linalg.norm(self.adv.flatten(), ord=2)
            print('success fitness[index]', fitness[index])

        # print(fitness)
        return fitness, fit




    def evaluater(self, x):

        neg_r_direction = (self.ori - self.adv).flatten()

        x = np.clip(x,0,1) * neg_r_direction



        with torch.no_grad():
            if torch.cuda.is_available():
                predict = self.model(torch.tensor(np.clip(np.tile(self.adv, (len(x), 1, 1, 1))
                                                          + np.reshape(x, (len(x),) + self.adv.shape) , 0.,
                                                          1.), dtype=torch.float32).cuda()).cpu()
            else:
                predict = self.model(torch.tensor(np.clip(np.tile(self.adv, (len(x), 1, 1, 1))
                                                          + np.reshape(x, (len(x),) + self.adv.shape) , 0.,
                                                          1.), dtype=torch.float32))
        p = np.copy(predict)
        q = np.zeros(p.shape)+0.5
        fit = p-q
        fit[:,self.target_label]=-fit[:,self.target_label]
        fit[np.where(fit<0)]=0
        fitness = np.sum(fit,axis=1)
        fitness = fitness[:, np.newaxis]

        print('ori',fitness)

        zero_fitness_indices = np.where(fitness == 0)[0]
        pop0 = x[zero_fitness_indices]

        for index in zero_fitness_indices:
            r = (np.reshape(x[index], self.adv.shape) + self.adv - self.ori).flatten()
            fitness[index] = np.linalg.norm(r, ord=2)
            # fitness[index] = np.linalg.norm(r, ord=2) / np.linalg.norm(self.adv.flatten(), ord=2)

        fitness0 = fitness[zero_fitness_indices]

        print('zero fitness[index]', fitness0)



        non_zero_fitness_indices = np.where(fitness != 0)[0]
        pop1 = x[non_zero_fitness_indices]
        for index in non_zero_fitness_indices:
            r = (np.reshape(x[index], self.adv.shape) + self.adv - self.ori).flatten()
            fitness[index] = np.linalg.norm(r, ord=2) + 10 * fitness[index]
            # fitness[index] = np.linalg.norm(r, ord=2) / np.linalg.norm(self.adv.flatten(), ord=2)

        fitness1 = fitness[non_zero_fitness_indices]
        print('non_zero fitness[index]', fitness1)


        # print(fitness)
        # return fitness, fit
        return pop0, fitness0, pop1, fitness1

def mating(pop,F):
    p2 = np.copy(pop)
    np.random.shuffle(p2)
    p3 = np.copy(p2)
    np.random.shuffle(p3)
    mutation = pop + F * (p2 - p3)
    return mutation

def matingr(pop,F):
    p2 = np.copy(pop)
    np.random.shuffle(p2)
    p3 = np.copy(p2)
    np.random.shuffle(p3)
    mutation = pop + 0.05 * (p2 + p3)
    return mutation

def mating_best_rand4(pop,fitness,F,gama):
    best = np.argmin(fitness)  # best�ǵ�ǰ��Сfitness�ı��
    mutation= np.copy(pop)
    popori = np.copy(pop)
    p2 = np.copy(pop)
    np.random.shuffle(p2)
    p3 = np.copy(p2)
    np.random.shuffle(p3)
    p4 = np.copy(p3)
    np.random.shuffle(p4)
    p5 = np.copy(p4)
    np.random.shuffle(p5)
    p6 = np.copy(p5)
    np.random.shuffle(p6)
    p7 = np.copy(p6)
    np.random.shuffle(p7)
    p8 = np.copy(p7)
    np.random.shuffle(p8)
    p9 = np.copy(p8)
    np.random.shuffle(p9)
    for i in range(len(pop)):
        mutation[i] = gama*pop[best] +(1-gama)* popori[i]+ (1-gama) * (p2[i] - p3[i]+ p4[i] - p5[i])
    return mutation

def mating_best(pop,fitness,F):
    best = np.argmin(fitness)  # best�ǵ�ǰ��Сfitness�ı��
    mutation= np.copy(pop)
    p2 = np.copy(pop)
    np.random.shuffle(p2)
    p3 = np.copy(p2)
    np.random.shuffle(p3)
    for i in range(len(pop)):
        mutation[i] = pop[best] + F * (p2[i]+p3[i])
    return mutation

def select(pop,fitness,fit,off,off_fitness,off_fit):
   new_pop = pop.copy()
   new_fitness = fitness.copy()
   new_fit = fit.copy()
   i=np.argwhere(fitness>off_fitness)
   new_pop[i] = off[i].copy()
   new_fitness[i] = off_fitness[i].copy()
   new_fit[i] = off_fit[i].copy()
   return new_pop ,new_fitness ,new_fit


def replace_bottom_10_percent(pop, fitness,fit, eps,length,problem):
    # ������Ӧ��ֵ����������
    # print(fitness)
    replace_count = len(pop) // 10
    sorted_indices = np.argsort(fitness.flatten())[-replace_count:]
    # print(sorted_indices)
    # ����Ҫ�滻�ĸ�������
    # ѡ��ֵ��С��10%���������
    replace_indices = sorted_indices[:replace_count]
    random = np.random.uniform(-eps, eps, size=(replace_count, length))
    for i,yi in enumerate(replace_indices):
        pop[yi] = random[i]
        # print(pop[yi].shape,random[i].shape)
    # fitness_, fit_replace_indices = problem.evaluate_mid(pop[replace_indices])
    # # print(fitness_)
    # fitness[replace_indices]=fitness_[:replace_count]
    # fit[replace_indices] = fit_replace_indices[:replace_count]
    # print(fitness)
    return pop,fitness,fit

def MLHS_Dy(pop_size, generation, length, model, image, target_label, eps, batch_size, gradient):
    # set seed
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
    np.random.seed(123)
    generation_save = np.zeros((10000,))
    problem = Problem2(model, image, image, target_label, eps, batch_size)
    pop = np.random.uniform(-eps, eps, size=(pop_size, length))
    # ͼ��ߴ���Ŷ�
    if (not (gradient is None)):
        pop[0] = np.reshape(np.sign(gradient), (length))
    max_eval = pop_size * generation
    eval_count = 0
    count = 0
    # ���� ������������pop_size
    fitness, fit = problem.evaluate_ini(pop)
    eval_count += pop_size
    # �ҳ������Ӧ��
    fitmin = np.min(fitness)
    generation_save[count] = fitmin
    # ��������Ӧ��==0 ���
    # print('fitmin',fitmin)
    if fitmin==0:
        print('success!!!!!!!!!!!!!!!!!!!!!!!!')
        return pop[np.where(fitness == 0)[0][0]], eval_count, generation_save[:count + 1], np.clip(np.reshape(pop[np.argmin(fitness)], image.shape) + image, 0, 1) ,1

    x_adv_t = np.clip(np.reshape(pop[np.argmin(fitness)], image.shape) + image, 0, 1)

    # ������������������µ�x_adv_t��������
    problem = Problem2(model,image, x_adv_t, target_label, eps, batch_size)

    # ��Ⱥʵ�ʱ仯��Ϊ
    pop_delta = (x_adv_t - image).flatten()

    pop, fitness, fit = replace_bottom_10_percent(pop, fitness, fit, eps, length, problem)

    pop = pop - pop_delta

    F = 0.5

    # ��û�����ɶԿ�����ʱ

    if (len(np.where(fitness == 0)[0]) == 0):
        while (eval_count < max_eval):

            gama = eval_count / (pop_size * generation)
            count += 1
            off = mating_best_rand4(pop, fitness, F, gama)
            # ������������pop_size�����Ӵ�����off
            off_fitness , off_fit = problem.evaluate_mid(off)
            eval_count += pop_size
            # ѡ��������Ⱥ����
            pop ,fitness ,fit = select (pop,fitness,fit,off,off_fitness,off_fit)
            fitmin = np.min(fitness)
            generation_save[count] = fitmin
            # print('fitmin', fitmin)
            # �����С��Ӧ��==0
            if (len(np.where(fitness == 0)[0]) != 0):
                x_adv_t1 = np.clip(np.reshape(pop[np.where(fitness == 0)[0][0]], x_adv_t.shape) + x_adv_t, 0, 1)
                print('success!!!!!!!!!!!!!!!!!!!!!!!!')
                return pop[np.where(fitness == 0)[0][0]], eval_count, generation_save[:count + 1], x_adv_t1 ,1
            pop_best = pop[np.argmin(fitness)]
            x_adv_t1 = np.clip(np.reshape(pop_best, x_adv_t.shape) + x_adv_t, 0, 1)
            problem = Problem2(model,image, x_adv_t1, target_label, eps, batch_size)
            pop_delta = (x_adv_t1 - x_adv_t).flatten()
            pop, fitness, fit = replace_bottom_10_percent(pop, fitness, fit, eps, length, problem)
            pop = pop - pop_delta
            x_adv_t = x_adv_t1
            # �������Ҫ
    if (len(np.where(fitness == 0)[0]) != 0):
        return pop[np.where(fitness == 0)[0][0]], eval_count, generation_save[:count + 1],x_adv_t ,1
    else:
        return pop[0], eval_count, generation_save[:count + 1],x_adv_t ,0


def binary_search_attack11(image, r, evaluate, tolerance=1e-5, max_iterations=100):
    # �� r ��С�� 0.4 �ĵ���Ϊ 0
    r[r < 0.4] = 0

    # ��ʼ�� low �� high
    low = 0
    high = np.max(r)

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)

        # �����Ŷ�ͼ��
        perturbed_r = np.resize(r, image.shape) * mid
        perturbed_image = image + perturbed_r

        # �����Ŷ����ͼ��
        is_adversarial = evaluate(perturbed_image)

        if is_adversarial:
            high = mid  # ����ǶԿ��Եģ����� low
        else:
            low = mid  # ������ǶԿ��Եģ����� high

        # ����Ƿ�����
        if high - low < tolerance:
            break

    return mid

def is_adversarial_f(model,perturbed_image,target):
    p_image = np.expand_dims(perturbed_image, axis=0)
    with torch.no_grad():
        if torch.cuda.is_available():
            predict = model(torch.tensor(p_image, dtype=torch.float32).cuda()).cpu()
        else:
            predict = model(torch.tensor(p_image, dtype=torch.float32))
    p = np.copy(predict)
    q = np.zeros(p.shape)+0.5
    fit = p-q
    fit[:,target]=-fit[:,target]
    fit[np.where(fit<0)]=0
    fitness = np.sum(fit,axis=1)
    fitness = fitness[:, np.newaxis]
    print(fitness == 0)
    print(fitness)
    if fitness == 0:
        return True
    else:
        return False

def get_top_n_elements(r, n=100):
    indices = np.argsort(r)[-n:][::-1]  # ��ȡ���� n ��Ԫ�ص�����
    top_elements = r[indices]            # ��ȡ��Ӧ��Ԫ��
    return indices, top_elements

def binary_search_attack(image, x_adv1, model,target, tolerance=1e-5, max_iterations=100):
    # �ҳ� r �д��� 0.4 ��Ԫ�ؼ�������
    r = (x_adv1 -image).flatten()
    # indices = np.where(r > 0.5)[0]
    print('this is r :', np.linalg.norm(r, ord=np.inf), np.linalg.norm(r, ord=2))
    p_r = np.copy(r)

    c=0
    while c < 5:
        c+=1

        indices, top_elements = get_top_n_elements(p_r, 100)

        low = 0
        high = np.max(top_elements)

        for _ in range(4):
            mid = 0.5 * (low + high)  # ���� mid

            # �� r ������ n ��Ԫ�ظ�ֵΪ mid
            p_r[indices] = mid

            # ������Ҫ�� r_modified ��������
            perturbed_image = np.clip(image + np.reshape(p_r, image.shape), 0., 1.)
            is_adversarial = is_adversarial_f(model,perturbed_image,target)

            if is_adversarial:
                high = mid  # ����ǶԿ��Եģ����� high
            else:
                low = mid  # ������ǶԿ��Եģ����� low

            # ����Ƿ�����
            if high - low < tolerance:
                break
        print('this is r after :', np.linalg.norm((perturbed_image-image).flatten(), ord=np.inf), np.linalg.norm((perturbed_image-image).flatten(), ord=2))

    return p_r

def DE_jianR(pop_size, generation, length, model, image,x_adv, target_label, eps, batch_size, gradient):
    # set seed
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
    np.random.seed(123)
    generation_save = np.zeros((10000,))
    pop_size = 20
    if generation>3:
        generation = 3
    ttt = 0.6
    problem = Problem2(model,image, x_adv, target_label, eps, batch_size)
    #(0,1)*r*(-1)
    pop = np.random.uniform(0., ttt, size=(pop_size, length)).astype(np.float32) * (image -x_adv).flatten()
    if (not (gradient is None)):
        pop[0] = np.reshape(np.sign(gradient), (length))
    max_eval = pop_size * generation
    eval_count = 0
    count = 0
    fitness, fit = problem.evaluate_adv(pop)

    if not np.all(fitness == 999) and np.sum(fitness != 999) >= 3:
        print('success ini', np.min(fitness))
    else:
        print('fail ini',np.min(fitness))

    eval_count += pop_size

    # �ҳ������Ӧ��
    fitmin = np.min(fitness)
    generation_save[count] = fitmin

    while np.all(fitness == 999) and eval_count <max_eval and count< 3 :

        ttt = 0.5*ttt
        print("All fail, now t is:", ttt)

        pop = np.random.uniform(0., ttt, size=(pop_size, length)).astype(np.float32) * (image - x_adv).flatten()
        # ���� ������������pop_size
        fitness, fit = problem.evaluate_adv(pop)
        eval_count += pop_size
        count+=1

        if not np.all(fitness == 999) and np.sum(fitness != 999) >= 3:
            print('success ini, l2 norm is:', np.min(fitness))
        fitmin = np.min(fitness)
        generation_save[count] = fitmin

    if eval_count >= max_eval or np.all(fitness == 999):
        return pop[np.argmin(fitness)], eval_count, generation_save[:count + 1], x_adv
    indices_to_keep = np.where(fitness < 999)[0]
    filtered_pop = pop[indices_to_keep]
    filtered_fitness = fitness[indices_to_keep]
    pop = filtered_pop
    fitness = filtered_fitness

    F = 0.5
    while (eval_count <max_eval and count< 3):
        count += 1
        off = matingr(pop,F)
        # off = mating(pop,F)
        off_fitness, off_fit = problem.evaluate_adv(off)
        eval_count += pop.shape[0]

        pop, fitness, fit = select(pop, fitness, fit, off, off_fitness, off_fit)
        fitmin = np.min(fitness)

        generation_save[count] = fitmin

        indices_to_keep = np.where(fitness < 999)[0]
        filtered_pop = pop[indices_to_keep]
        filtered_fitness = fitness[indices_to_keep]
        pop = filtered_pop
        fitness = filtered_fitness

    print('final l2 norm:', np.min(fitness))
    x_adv_ = np.clip(np.reshape(pop[np.argmin(fitness)],  image.shape)+x_adv,0,1)
    return pop[np.argmin(fitness)], eval_count, generation_save[:count + 1], x_adv_
