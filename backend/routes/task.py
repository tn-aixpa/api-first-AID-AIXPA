import json
import urllib
from typing import Annotated, Optional
import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlmodel import Session, select

from routes.file import get_file_content
from routes.login import get_current_user, user_must_be_admin
from routes.project import check_manage_project
from sql.crud import get_object_by_id
from sql.database import get_db
from sql.models import Task, User, TaskOut, TaskCreate, TaskUserLink, File, TaskFileLink, Actor, ActorCreate, \
    TaskStartType, TaskInsideType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/import")
async def call_import(
        db: Annotated[Session, Depends(get_db)],
        is_admin: Annotated[bool, Depends(user_must_be_admin)],
        user: Annotated[User, Depends(get_current_user)],
        project_id: int,
        file: UploadFile
) -> list[TaskOut]:
    json_data = json.load(file.file)
    db_project = check_manage_project(db, project_id, user)

    return_tasks = []

    for d in json_data:
        task_actors = {}
        task_files = set()
        for turn in d['data']:
            speaker = turn['speaker']
            ground = bool(turn['ground']) and len(turn['ground']) > 0
            if ground:
                for g in turn['ground']:
                    task_files.add(g['file_id'])
                pass
            if speaker not in task_actors:
                task_actors[speaker] = ActorCreate(label=speaker, name=d['roles'][speaker], ground=ground)
            else:
                task_actors[speaker].ground = task_actors[speaker].ground or ground

        task_users = set()
        for u in d['users']:
            this_user = db.query(User).filter(User.username == u).first()
            if not this_user:
                raise HTTPException(status_code=400, detail=f"User {u} not found")
            task_users.add(this_user.id)

        # TODO: Duplicated code, should be a function
        allowed_files = set()
        files = []
        for file in db_project.files:
            allowed_files.add(file.id)
        for fid in task_files:
            if fid not in allowed_files:
                raise HTTPException(status_code=400, detail=f"File {fid} is not allowed in project {project_id}")
            files.append(fid)
        allowed_users = set()
        users = []
        for user in db_project.users:
            allowed_users.add(user.user_id)
        for uid in task_users:
            if uid not in allowed_users:
                raise HTTPException(status_code=400, detail=f"User {uid} is not allowed in project {project_id}")
            users.append(uid)

        # TODO: Duplicated code (again), should be a function
        obj_to_add = []
        # file_contents = []
        # file_ids = []

        task_info = {
            "name": d['task_name'],
            "start_type": TaskStartType.pre_compiled,
            "inside_type": TaskInsideType.clean,
            "language": d['language'],
            "meta": {
                "new_annotation_data": d['data']
            },
        }

        db_task = Task.model_validate(task_info, update={"project_id": project_id})
        obj_to_add.append(db_task)

        if users is not None:
            for user in users:
                db_user = get_object_by_id(db, user, User)
                if not db_user:
                    raise HTTPException(status_code=404, detail=f"User {user} not found")
                db_obj = TaskUserLink(task=db_task, user=db_user)
                obj_to_add.append(db_obj)
        if files is not None:
            for file in files:
                db_file = get_object_by_id(db, file, File)
                if not db_file:
                    raise HTTPException(status_code=400, detail=f"File {file} not found")
                # file_contents.append(get_file_content(db_file))
                # file_ids.append(file)
                db_obj = TaskFileLink(task=db_task, file=db_file)
                obj_to_add.append(db_obj)
        actors_order = 0
        for k in task_actors:
            a = task_actors[k]
            db_actor = Actor(**a.__dict__, task=db_task, ord=actors_order)
            actors_order += 1
            db.add(db_actor)

        for o in obj_to_add:
            db.add(o)

        db.commit()
        db.refresh(db_task)
        return_tasks.append(db_task)
    return return_tasks


@router.post("/", status_code=201)
async def call_create_task(
        db: Annotated[Session, Depends(get_db)],
        user: Annotated[User, Depends(get_current_user)],
        project_id: int,
        task: TaskCreate,
) -> Optional[TaskOut]:
    tryout = task.tryout
    db_project = check_manage_project(db, project_id, user)
    allowed_files = set()
    files = []
    for file in db_project.files:
        allowed_files.add(file.id)
    for fid in task.files_list:
        if fid not in allowed_files:
            raise HTTPException(status_code=400, detail=f"File {fid} is not allowed in project {project_id}")
        files.append(fid)
    allowed_users = set()
    users = []
    for user in db_project.users:
        allowed_users.add(user.user_id)
    for uid in task.users_list:
        if uid not in allowed_users:
            raise HTTPException(status_code=400, detail=f"User {uid} is not allowed in project {project_id}")
        users.append(uid)

    obj_to_add = []
    file_contents = []
    file_ids = []

    db_task = Task.model_validate(task, update={"project_id": project_id})
    obj_to_add.append(db_task)

    if users is not None:
        for user in users:
            db_user = get_object_by_id(db, user, User)
            if not db_user:
                raise HTTPException(status_code=404, detail=f"User {user} not found")
            db_obj = TaskUserLink(task=db_task, user=db_user)
            obj_to_add.append(db_obj)
    if files is not None:
        for file in files:
            db_file = get_object_by_id(db, file, File)
            if not db_file:
                raise HTTPException(status_code=400, detail=f"File {file} not found")
            file_contents.append(get_file_content(db_file))
            file_ids.append(file)
            db_obj = TaskFileLink(task=db_task, file=db_file)
            obj_to_add.append(db_obj)

    if len(file_contents) == 0:
        raise HTTPException(status_code=400, detail=f"No files selected")

    new_annotation_data = []

    if task.start_type != "empty":
        if "start_type_url" not in task.meta or not task.meta["start_type_url"]:
            raise HTTPException(status_code=400, detail=f"Empty start_type_url")
        if "start_type_method" not in task.meta or not task.meta["start_type_method"]:
            raise HTTPException(status_code=400, detail=f"Empty start_type_method")
        allowed_methods = {}
        headers = {"Content-Type": "application/json"}
        if task.meta['start_type_header_key']:
            headers[task.meta['start_type_header_key']] = task.meta['start_type_header_value']
        url = task.meta['start_type_url']
        try:
            req = urllib.request.Request(url, headers=headers)
            contents = urllib.request.urlopen(req).read()
            api_info = json.loads(contents.decode())
            for m in api_info:
                allowed_methods[m['generation_method']] = m['roles']
            if task.meta['start_type_method'] not in allowed_methods:
                raise HTTPException(status_code=400, detail=f"Unknown method {task.meta['start_type_method']}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        roles = allowed_methods[task.meta['start_type_method']]
        for role in roles:
            if "answers" not in role:
                role['answers'] = 1
        actors_list = []
        for a in task.actors_list:
            actors_list.append(a.__dict__)
        if actors_list != roles:
            raise HTTPException(status_code=400, detail=f"Invalid list of roles")

        # Get the first dialogue
        # TODO: get the ground information from interface
        post_data = {}
        post_data['generation_mode'] = task.meta['start_type_method']
        # post_data['num_turns'] = 10
        post_data['documents'] = file_contents
        post_data['ground_required'] = {"speaker_1": False, "speaker_2": True}

        # https://stackoverflow.com/questions/62384020/python-3-7-urllib-request-doesnt-follow-redirect-url
        req = urllib.request.Request(url, json.dumps(post_data).encode(), headers=headers)
        try:
            try:
                urlopen = urllib.request.urlopen(req)
            except urllib.error.HTTPError as e:
                if e.status != 307:
                    raise  # not a status code that can be handled here
                redirected_url = urllib.parse.urljoin(url, e.headers['Location'])
                req = urllib.request.Request(redirected_url, json.dumps(post_data).encode(),
                                             headers=headers)
                urlopen = urllib.request.urlopen(req)
                logger.info('Redirected -> %s' % redirected_url)  # the original redirected url
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Unable to run script: {str(e)}")
        response = urlopen.read()

        json_data = json.loads(response.decode())
        for t in json_data:
            new_item = {}
            new_item['speaker'] = t['speaker']
            new_item['text'] = t['turn_text']
            new_item['ground'] = []
            for g in t['ground']:
                new_ground = {}
                new_ground['text'] = g['text']
                new_ground['file_id'] = file_ids[g['file_index']]
                new_ground['offset_start'] = g['offset_start']
                new_ground['offset_end'] = g['offset_end']
                new_item['ground'].append(new_ground)
            new_annotation_data.append(new_item)

    db_task.meta['new_annotation_data'] = new_annotation_data

    # TODO: check and insert actors when inside_type is choice

    actors = set()
    for a in task.actors_list:
        if a.name in actors:
            raise HTTPException(status_code=400, detail=f"Actor {a.name} is duplicated")
        actors.add(a.name)

    if tryout:
        return None

    ord = 0
    for a in task.actors_list:
        db_actor = Actor(**a.__dict__, task=db_task, ord=ord)
        ord += 1
        db.add(db_actor)

    for o in obj_to_add:
        db.add(o)

    db.commit()
    db.refresh(db_task)
    return db_task

    # db_user = crud.get_user_by_email(db, email=user.email)
    # if db_user:
    #     raise HTTPException(status_code=400, detail="Email already registered")
    # try:
    #     return crud.create_user(db=db, user=user)
    # except Exception as e:
    #     raise HTTPException(status_code=400, detail=str(e))


@router.get("/{task_id}")
async def call_task_info(
        db: Annotated[Session, Depends(get_db)],
        user: Annotated[User, Depends(get_current_user)],
        project_id: int,
        task_id: int
) -> TaskOut:
    db_task = get_object_by_id(db, task_id, Task)
    if db_task.project_id != project_id:
        raise HTTPException(status_code=400, detail=f"Task {task_id} does not belong to project {project_id}")
    (db_project, is_manager) = check_manage_project(db, db_task.project_id, user, True)
    if not is_manager:
        stmt = select(Task, TaskUserLink).where(Task.id == task_id, TaskUserLink.user_id == user.id)
        result = db.exec(stmt).first()
        if not result:
            raise HTTPException(status_code=400, detail=f"User {user.username} cannot access task {task_id}")
    return db_task


@router.patch("/{task_id}/deactivate")
async def deactivate_task(
        db: Annotated[Session, Depends(get_db)],
        user: Annotated[User, Depends(get_current_user)],
        project_id: int,
        task_id: int
) -> TaskOut:
    # TODO: check stuff
    db_task = get_object_by_id(db, task_id, Task)
    if db_task.project_id != project_id:
        raise HTTPException(status_code=400, detail=f"Task {task_id} does not belong to project {project_id}")
    db_task.is_active = False
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task


@router.patch("/{task_id}/activate")
async def activate_task(
        db: Annotated[Session, Depends(get_db)],
        user: Annotated[User, Depends(get_current_user)],
        project_id: int,
        task_id: int
) -> TaskOut:
    # TODO: check stuff
    db_task = get_object_by_id(db, task_id, Task)
    if db_task.project_id != project_id:
        raise HTTPException(status_code=400, detail=f"Task {task_id} does not belong to project {project_id}")
    db_task.is_active = True
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task


@router.delete("/{task_id}", status_code=204)
async def delete_task(
        db: Annotated[Session, Depends(get_db)],
        user: Annotated[User, Depends(get_current_user)],
        project_id: int,
        task_id: int
):
    # TODO: check stuff
    db_task = get_object_by_id(db, task_id, Task)
    db_task.is_deleted = True
    db.add(db_task)
    db.commit()
