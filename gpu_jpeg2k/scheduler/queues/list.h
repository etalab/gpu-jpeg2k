/**
 * @file list.h
 *
 * @author Milosz Ciznicki
 */

#ifndef LIST_H_
#define LIST_H_

#include <stdlib.h>

/** Generates a new type for list of elements */
#define LIST_TYPE(ENAME, DECL) \
  LIST_DECLARE_TYPE(ENAME) \
  LIST_CREATE_TYPE(ENAME, DECL)

/** Forward type declaration for lists */
#define LIST_DECLARE_TYPE(ENAME) \
  /** automatic type: ENAME##_list_t is a list of ENAME##_t */ \
  typedef struct ENAME##_list_s* ENAME##_list_t; \
  /** automatic type: defines ENAME##_t */ \
  typedef struct ENAME##_s* ENAME##_t; \
  /** automatic type: ENAME##_itor_t is an iterator on lists of ENAME##_t */ \
  typedef ENAME##_t ENAME##_itor_t;

/** The effective type declaration for lists */
#define LIST_CREATE_TYPE(ENAME, DECL) \
  /** from automatic type: ENAME##_t */ \
  struct ENAME##_s \
  { \
    struct ENAME##_s* _prev; /**< @internal previous cell */ \
    struct ENAME##_s* _next; /**< @internal next cell */ \
    DECL \
  }; \
   \
  struct ENAME##_list_s \
  { \
    struct ENAME##_s* _head; /**< @internal head of the list */ \
    struct ENAME##_s* _tail; /**< @internal tail of the list */ \
  }; \
  static inline ENAME##_t ENAME##_new(void) \
    { ENAME##_t e = (ENAME##_t)malloc(sizeof(struct ENAME##_s)); \
      e->_next = NULL; e->_prev = NULL; return e; } \
  static inline void ENAME##_delete(ENAME##_t e) \
    { free(e); } \
  static inline void ENAME##_list_push_front(ENAME##_list_t l, ENAME##_t e) \
    { if(l->_tail == NULL) l->_tail = e; else l->_head->_prev = e; \
      e->_prev = NULL; e->_next = l->_head; l->_head = e; } \
  static inline void ENAME##_list_push_back(ENAME##_list_t l, ENAME##_t e) \
    { if(l->_head == NULL) l->_head = e; else l->_tail->_next = e; \
      e->_next = NULL; e->_prev = l->_tail; l->_tail = e; } \
  static inline ENAME##_t ENAME##_list_front(ENAME##_list_t l) \
    { return l->_head; } \
  static inline ENAME##_t ENAME##_list_back(ENAME##_list_t l) \
    { return l->_tail; } \
  static inline ENAME##_list_t ENAME##_list_new(void) \
    { ENAME##_list_t l; l=(ENAME##_list_t)malloc(sizeof(struct ENAME##_list_s)); \
      l->_head=NULL; l->_tail=l->_head; return l; } \
  static inline int ENAME##_list_empty(ENAME##_list_t l) \
    { return (l->_head == NULL); } \
  static inline void ENAME##_list_delete(ENAME##_list_t l) \
    { free(l); } \
  static inline void ENAME##_list_erase(ENAME##_list_t l, ENAME##_t c) \
    { ENAME##_t p = c->_prev; if(p) p->_next = c->_next; else l->_head = c->_next; \
      if(c->_next) c->_next->_prev = p; else l->_tail = p; } \
  static inline ENAME##_t ENAME##_list_pop_front(ENAME##_list_t l) \
    { ENAME##_t e = ENAME##_list_front(l); \
      ENAME##_list_erase(l, e); return e; } \
  static inline ENAME##_t ENAME##_list_pop_back(ENAME##_list_t l) \
    { ENAME##_t e = ENAME##_list_back(l); \
      ENAME##_list_erase(l, e); return e; } \
  static inline ENAME##_itor_t ENAME##_list_begin(ENAME##_list_t l) \
    { return l->_head; } \
  static inline ENAME##_itor_t ENAME##_list_end(ENAME##_list_t l __attribute__ ((unused))) \
    { return NULL; } \
  static inline ENAME##_itor_t ENAME##_list_next(ENAME##_itor_t i) \
    { return i->_next; } \
  static inline int ENAME##_list_size(ENAME##_list_t l) \
    { ENAME##_itor_t i=l->_head; int k=0; while(i!=NULL){k++;i=i->_next;} return k; } \
  static inline int ENAME##_list_check(ENAME##_list_t l) \
    { ENAME##_itor_t i=l->_head; while(i) \
    { if ((i->_next == NULL) && i != l->_tail) return 0; \
      if (i->_next == i) return 0; \
      i=i->_next;} return 1; }

#endif /* LIST_H_ */
