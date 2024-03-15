import { Button, Checkbox, FormControlLabel, FormGroup, TextField } from '@mui/material';
import React, { Component } from 'react';
import Modal from 'react-bootstrap-modal';
import APIServices from '../../API/apiservices';
import DataTableComponent from '../../Component/DataTableComponent';
import TopMenuBar from '../../Component/TopMenuBar';
import "./index.scss";


const apiServices = new APIServices();
export class AlertDetails extends Component {

    constructor(props) {
        super(props)
        this.bcData = []
        this.state = {
            region: null,
            country: null,
            city: null,
            route: null,
            cabinValue: null,
            gettingMonth: null,
            gettingYear: null,
            rowData: [],
            headers: [],
            loading: false,
            rowDataMarketShare: [],
            headersMarketShare: [],
            loadingMS: false,
            rowDataInfare: [],
            headersInfare: [],
            loadingInfare: false,
            rowDataAvail: [],
            headersAvail: [],
            loadingAvail: false,
            message: null,
            messageModalVisible: false,
            action: "Take",
            actionId: null,
            summaryData: [],
            flag: null,
        }
    }

    componentDidMount() {
        this.getFiltersValue()
    }

    getFiltersValue = () => {
        this.bcData = []

        const region = this.getItemFromLS('RegionSelected')
        const country = this.getItemFromLS('CountrySelected')
        const city = this.getItemFromLS('CitySelected')
        const route = this.getItemFromLS('ODSelected')
        const cabinValue = this.getItemFromLS('SelectedCabin')
        let rangeValue = JSON.parse(window.localStorage.getItem('rangeValue'))


        this.setState({
            region, country, city, route, cabinValue,
            gettingMonth: window.monthNumToName(rangeValue.from.month),
            gettingYear: rangeValue.from.year,
        }, () => this.getAlertDetailsData())


        this.bcData.push({ val: region, title: "Region" })
        this.bcData.push({ val: country, title: "Country" })
        this.bcData.push({ val: city, title: "City" })
        this.bcData.push({ val: route, title: "Route" })
        this.bcData.push({ val: cabinValue, title: "Cabin" })
    }

    getAlertSummary = (alertId) => {
        apiServices.getAlertSummary(alertId).then((data) => {
            this.setState({ summaryData: data })
        })
    }

    getAlertDetailsData = () => {
        const { gettingYear, gettingMonth, region, country, city, route, cabinValue } = this.state;
        apiServices.getAlertDetails(gettingYear, gettingMonth, region, country, city, route, cabinValue).then((result) => {
            if (result) {
                const columnName = result.columnName;
                const rowData = result.rowData;
                this.setState({
                    actionId: result.id,
                    headers: columnName[0],
                    rowData: rowData[0],
                    headersMarketShare: columnName[1],
                    rowDataMarketShare: rowData[1],
                    headersInfare: columnName[2],
                    rowDataInfare: rowData[2],
                    headersAvail: columnName[3],
                    rowDataAvail: rowData[3],
                    flag: result.flag
                }, () => this.getAlertSummary(this.state.actionId))
            }
        });
    }

    handleChange = (e) => {
        this.setState({ message: e.target.value })
    }

    showModal = () => {
        this.setState({ messageModalVisible: true })
    }

    closeModal = () => {
        this.setState({ messageModalVisible: false })
    }

    setAction = (action) => {
        this.setState({ action })
    }

    postAlertDetails = () => {
        const { region, country, city, route, cabinValue, action, actionId, message } = this.state;
        apiServices.postAlertDetails(region, country, city, route, cabinValue, action, actionId, message).then(() => {
            this.setState({ message: "", messageModalVisible: false })
        })
    }

    getLink = (type) => {
        const { region, country, city, route } = this.state;
        if (type === 'Region') {
            return `/alertsNew?Region=${encodeURIComponent(region)}`
        } else if (type === 'Country') {
            return `/alertsNew?Region=${encodeURIComponent(region)}&Country=${country}`
        } else if (type === 'City') {
            return `/alertsNew?Region=${encodeURIComponent(region)}&Country=${country}&POS=${city}`
        } else if (type === 'Route') {
            return `/alertsNew?Region=${encodeURIComponent(region)}&Country=${country}&POS=${city}&${encodeURIComponent('O&D')}=${route}`
        }
    }

    maybeClearLS = () => {
        if (this.state.region === "*") {
            localStorage.removeItem("CitySelected")
            localStorage.removeItem("ODSelected")
            localStorage.removeItem("SelectedCabin")
        }
    }

    render() {
        const { region, country, city, flag } = this.state;
        return (
            <div style={{ fontSize: "16px", height: "100vh", display: "flex", flexDirection: "column" }}>
                <TopMenuBar {...this.props} />
                <div className="row">
                    <div className="col-md-12 col-sm-12 col-xs-12 top">
                        <div className="navdesign" style={{ marginTop: '0px' }}>
                            <div className="col-md-7 col-sm-7 col-xs-7 toggle1">
                                <a href={region === "*" ? "/alertsNew" : `/alertsNew?Region=${encodeURIComponent(region)}&Country=${country}&POS=${city}`} style={{ display: "inline-block", color: "white" }} onClick={this.maybeClearLS}><h3>Alert Module</h3></a>
                                <section style={{ display: "inline-block" }}>
                                    <nav>
                                        <ol className="cd-breadcrumb" style={{ listStyle: "none" }}>
                                            <li>Network</li>
                                            {this.state.firstLoadList ? "" : this.bcData.map((item) =>
                                                item.val === "*" ? "" :
                                                    <li style={{ textDecoration: "none", display: "inline-block" }} id={item.val} title={`${item.title} : ${item.val}`} key={item.title}>
                                                        <a style={{ textDecoration: "none", color: "white" }} href={this.getLink(item.title)}> {` > ${item.val}`}</a>
                                                    </li>
                                            )}
                                        </ol>
                                    </nav>
                                </section>
                            </div>
                        </div>
                    </div>
                </div>
                <div className='row container-details' >
                    <div className='col tables' style={{ overflowY: "scroll" }}>
                        <DataTableComponent
                            rowData={this.state.rowData}
                            columnDefs={this.state.headers}
                            loading={this.state.loading2}
                        />

                        <DataTableComponent
                            rowData={this.state.rowDataMarketShare}
                            columnDefs={this.state.headersMarketShare}
                            autoHeight='autoHeight'
                            loading={this.state.loadingMS}
                        />
                        <DataTableComponent
                            rowData={this.state.rowDataInfare}
                            columnDefs={this.state.headersInfare}
                            autoHeight='autoHeight'
                            loading={this.state.loadingInfare}
                        />
                        <DataTableComponent
                            autoHeight='autoHeight'
                            rowData={this.state.rowDataAvail}
                            columnDefs={this.state.headersAvail}
                            loading={this.state.loadingAvail}
                        />

                        {flag === "No" && (<FormGroup sx={{ justifyContent: "flex-start", gap: "10px", padding: "20px" }} row style={{ fontSize: "12px" }}>
                            <FormControlLabel control={<Checkbox sx={{
                                color: "gray",
                            }} checked={this.state.action === "Take"} onChange={(e) => { if (e.target.checked) this.setAction("Take") }} />} label="Action Taken" />
                            <FormControlLabel control={<Checkbox sx={{
                                color: "gray",
                            }} checked={this.state.action === "Reject"} onChange={(e) => { if (e.target.checked) this.setAction("Reject") }} />} label="Rejected" />
                            <Button variant='contained' onClick={() => this.showModal()}>Submit</Button>
                        </FormGroup>)}
                    </div>
                    <div style={{ borderRight: "1px solid gray", margin: "30px 0" }}></div>
                    <div className='col' style={{ margin: "12px 20px" }}>
                        <h3>Alert Summary:</h3>
                        {this.renderSummary(this.state.summaryData)}
                    </div>
                </div>
                <Modal
                    show={this.state.messageModalVisible}
                    onHide={() => this.closeModal()}
                    aria-labelledby="ModalHeader"
                >
                    <Modal.Header closeButton>
                        <Modal.Title id='ModalHeader'>Add a message</Modal.Title>
                    </Modal.Header>
                    <Modal.Body>
                        <TextField
                            id="outlined-multiline-flexible"
                            label="Message"
                            multiline
                            rows={4}
                            fullWidth
                            value={this.state.message}
                            onChange={this.handleChange}
                        />
                        <div style={{ marginTop: "1rem", display: "flex", gap: "1rem", justifyContent: "flex-end" }}>
                            <Button variant='contained' onClick={this.postAlertDetails}>Ok</Button>
                            <Button variant='outlined' onClick={this.closeModal}>Cancel</Button>
                        </div>
                    </Modal.Body>
                </Modal>
            </div >
        )
    }

    getValueText(value) {
        return value > 0 ? "above" : "below"
    }

    convertZeroValueToBlank(value) {
        let convertedValue =
            window.numberFormat(value) === 0 ? "0" : window.numberFormat(value);
        return convertedValue;
    }

    renderValue(value) {
        return (<span title={window.numberWithCommas(value)} > {this.convertZeroValueToBlank(value)}</span>)
    }

    renderSummary(summaryData) {
        return summaryData.map((item) => {
            return (
                <div style={{ fontSize: "18px", fontWeight: 300, margin: "20px 0px" }}>
                    <div style={{ display: "flex", gap: "5px" }}>
                        <p><b>{item.alertDates} </b></p>
                        <p>
                            : Revenue is {this.getValueText(item.Rev_TGT)} target by {this.renderValue(item.Rev_TGT)} {this.renderArrowIcon(item.Rev_TGT)},
                            Booking is {this.getValueText(item.Book_TGT)} target by {this.renderValue(item.Book_TGT)} {this.renderArrowIcon(item.Book_TGT)},
                            <br />
                            &nbsp;&nbsp;Passenger is {this.getValueText(item.Pax_TGT)} target by {this.renderValue(item.Pax_TGT)} {this.renderArrowIcon(item.Pax_TGT)},
                            Average Fare is {this.getValueText(item.Avg_TGT)} target by {this.renderValue(item.Avg_TGT)} {this.renderArrowIcon(item.Avg_TGT)}
                            <br />
                            <br />
                            : Action for {item.alert_action}
                        </p>
                    </div>
                    {this.state.flag === "Yes" && (<div>
                        {item.commentTime} : Actioned by {item.userName ?? "User"}
                        <br />
                        <b>Comment :</b> {item.commentText ?? "Empty"}
                    </div>)}
                </div>
            )
        })
    }

    getItemFromLS(name) {
        const value = localStorage.getItem(name)
        return value === null || value === 'Null' || value === '' ? '*' : JSON.parse(value)
    }

    renderArrowIcon(value) {
        if (value > 0) {
            return <i className='fa fa-arrow-up'></i>
        } else if (value === 0) {
            return null
        } else {
            return <i className='fa fa-arrow-down'></i>
        }
    }
}

export default AlertDetails